"""Partially pooled fixture context inside one coherent tool-return model.

The reference q(Z_results,Z_xg | Y,p0) is retained as the population prior.
Past matches sharing the competition and either same-role team provide local
counts. Dirichlet shrinkage toward the reference, followed by held-out choice
of context/evidence strengths, avoids treating an identity match as certainty.

The initial joint is built once from free fixture context. Subsequent revealed
contents only condition that joint: the context kernel is never reset using a
posterior, a hypothetical return, an unqueried feature, or an evaluation label.
This is a statistical response-model extension, not a new VoI theorem.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .conservative_joint import (
    ACQUIRED_SETS, ConservativeJointReturnPredictor, _digest,
    _restore_reference, minimize_strength,
)
from .experiment import outcome, state_after
from .joint_returns import (
    JointReturnPredictor, RESULT_CATEGORIES, XG_CATEGORIES, _probabilities,
)
from .state import parse_time

VERSION = "contextual-joint-returns-v1"
CONTEXT_WEIGHTS = (0., .5, 1.)
FIXTURE_FIELDS = ("match_id", "competition_id", "home_team_id", "away_team_id",
                  "prediction_cutoff", "kickoff", "result_available_at",
                  "fixture_available_at", "information_mode")


def _unit(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(name + " must be finite and in [0, 1]")
    return float(value)


class ContextualJointReturnPredictor(JointReturnPredictor):
    """Same finite return contract; only the frozen initial likelihood changes.

    ``fixture_rows`` may contain training features, but only FIXTURE_FIELDS for
    reference training IDs are read. Runtime never receives these full rows.
    Use ``fit_context`` for past-only selection. Manual parameters are declared
    mechanism experiments, and must not be described as calibrated estimates.
    """

    def __init__(self, reference, fixture_rows, *, context_weight=1., strength=1.,
                 selection=None):
        if type(reference) is not JointReturnPredictor or not reference.model_version:
            raise ValueError("A frozen original JointReturnPredictor is required")
        context_weight = _unit(context_weight, "Context weight")
        strength = _unit(strength, "Evidence strength")
        payload = deepcopy({**reference._payload(), "model_version": reference.model_version})
        snapshot = _restore_reference(payload)
        super().__init__(snapshot.base_predictor, snapshot.bandwidth, snapshot.dirichlet_mass)
        self._reference_payload = payload
        self.context_weight, self.evidence_strength = context_weight, strength
        self._selection = deepcopy(selection)
        self.quantiles, self.records, self.metadata = snapshot.quantiles, snapshot.records, snapshot.metadata
        by_id = {r["match_id"]: r for r in self.records}
        self._fixture_rows, seen = [], set()
        for row in fixture_rows:
            mid = str(row["match_id"])
            if mid not in by_id:
                continue
            if mid in seen:
                raise ValueError("Duplicate reference fixture context")
            seen.add(mid)
            # Do not serialize, iterate over, or inspect any other row value.
            projection = {k: row[k] for k in FIXTURE_FIELDS}
            projection["match_id"] = mid
            for key in ("competition_id", "home_team_id", "away_team_id"):
                if not isinstance(projection[key], str) or not projection[key]:
                    raise ValueError("Explicit fixture identity required")
            for key in ("prediction_cutoff", "kickoff", "result_available_at"):
                if parse_time(projection[key]) != parse_time(by_id[mid][key]):
                    raise ValueError("Fixture context/reference chronology mismatch")
            if (projection["information_mode"] != self.metadata["information_mode"] or
                    parse_time(projection["fixture_available_at"]) > parse_time(projection["prediction_cutoff"])):
                raise ValueError("Fixture context unavailable at reference prediction time")
            self._fixture_rows.append(projection)
        if seen != set(by_id):
            raise ValueError("Every reference training match needs a public fixture context")
        self._fixture_rows.sort(key=lambda r: r["match_id"])
        fixtures = {r["match_id"]: r for r in self._fixture_rows}
        self._record_contexts = [self._context(fixtures[r["match_id"]]) for r in self.records]
        if any(None in context for context in self._record_contexts):
            raise ValueError("Reference fixture context has an unmapped identity")
        self._context_cache, self._adaptation_ids = {}, set()
        if selection is not None:
            selected = selection.get("selected", {})
            if (selection.get("schema") != VERSION or
                    selection.get("reference_payload_sha256") != _digest(payload) or
                    selection.get("fixture_context_sha256") != _digest(self._fixture_rows) or
                    selected.get("context_weight") != context_weight or
                    selected.get("strength") != strength):
                raise ValueError("Context selection is not bound to this fitted model")
            if parse_time(selection["as_of"]) <= parse_time(self.metadata["training_as_of"]):
                raise ValueError("Context adaptation must follow reference fitting")
            self._adaptation_ids = set(selection["match_ids"])
            trained = set(self.metadata["training_match_ids"]) | set(self.metadata["base_training_match_ids"])
            if self._adaptation_ids & trained:
                raise ValueError("Context adaptation reused a reference training match")
            self.metadata["training_as_of"] = self.metadata["trained_as_of"] = selection["as_of"]
        self.metadata.update(
            probability_model="p0(Y|H) * coherent_strength(q_context(Z_results,Z_xg|Y,p0,fixture))",
            q_context="free exact competition and home/away-role team IDs, plus past forward OOF p0",
            sufficiency_assumption="finite return summaries and exact identity pooling; unproven in football",
            context_adaptation={
                "context_weight": context_weight, "evidence_strength": strength,
                "population_prior_mass_per_outcome": self.dirichlet_mass,
                "reference_model_version": reference.model_version,
                "fixture_context_sha256": _digest(self._fixture_rows),
                "selection_kind": "chronological_same_version_heldout" if selection else "declared_mechanism_only",
                "selection_match_ids": sorted(self._adaptation_ids),
                "context_rule": "same_competition * mean(same_home_role_team, same_away_role_team)",
                "context_backoff": "zero local weight returns the population likelihood exactly",
                "calibration_status": "not_established",
            })
        self.model_version = self.version = VERSION + ":" + _digest(self._payload())[:20]

    def _context(self, base):
        aliases = self.base_predictor.crosswalk["aliases"]
        return (aliases["competition"].get(base["competition_id"]),
                aliases["team"].get(base["home_team_id"]),
                aliases["team"].get(base["away_team_id"]))

    @property
    def selection_report(self):
        return deepcopy(self._selection)

    def fit(self, *args, **kwargs):
        raise TypeError("Use fit_context with a frozen reference and past held-out rows")

    def _visible(self, state, *, hypothetical=False):
        clean, observations = super()._visible(state, hypothetical=hypothetical)
        if clean["match_id"] in self._adaptation_ids:
            raise ValueError("Context adaptation match cannot be evaluated as held out")
        return clean, observations

    def _context_q(self, p0, context, *, condition_on_p0=True):
        key = (tuple(p0), tuple(context), bool(condition_on_p0))
        if key in self._context_cache:
            return self._context_cache[key]
        population, audit = super()._q(p0, condition_on_p0=condition_on_p0)
        arrays = self._arrays
        weights = (np.exp(-np.sum((arrays["priors"]-np.asarray(p0))**2, axis=1) /
                          (2*self.bandwidth**2)) if condition_on_p0 else np.ones(len(self.records)))
        overlaps = np.asarray([
            float(context[0] is not None and old[0] == context[0]) *
            (float(context[1] is not None and old[1] == context[1]) +
             float(context[2] is not None and old[2] == context[2])) / 2.
            for old in self._record_contexts])
        local_weights = weights*overlaps
        counts = np.zeros_like(population)
        np.add.at(counts, (arrays["y"], arrays["r"], arrays["x"]), local_weights)
        class_mass = counts.sum(axis=(1, 2))
        local = ((counts + self.dirichlet_mass*population) /
                 (class_mass+self.dirichlet_mass)[:, None, None])
        a = self.context_weight
        contextual = (1-a)*population + a*local
        marginal = np.sum(np.asarray(p0)[:, None, None]*contextual, axis=0)
        s = self.evidence_strength
        mixed = s*contextual + (1-s)*marginal[None, :, :]
        mass, square = float(local_weights.sum()), float(np.square(local_weights).sum())
        details = {**audit, "context_weight": a, "evidence_strength": s,
                   "local_effective_matches": mass*mass/square if square else 0.,
                   "local_matching_training_matches": int(np.count_nonzero(local_weights)),
                   "local_class_kernel_weight": {str(y): float(class_mass[y]) for y in range(3)},
                   "local_posterior_weight_per_outcome": {
                       str(y): float(a*class_mass[y]/(class_mass[y]+self.dirichlet_mass)) for y in range(3)},
                   "local_backoff": "population" if not mass else "partial_pooling",
                   "population_prior_mass_per_outcome": self.dirichlet_mass,
                   "conditioning_scope": "same_initial_joint_before_any_observation",
                   "calibration_status": "not_established"}
        self._context_cache[key] = mixed, details
        return mixed, details

    def joint_distribution(self, state, *, hypothetical=False, condition_on_p0=True):
        clean, observations = self._visible(state, hypothetical=hypothetical)
        prior = self._free_prior(clean)
        p0 = _probabilities(prior["probabilities"])
        q, audit = self._context_q(p0, self._context(clean["base"]), condition_on_p0=condition_on_p0)
        joint = p0[:, None, None]*q
        if "results_view" in observations:
            joint[:, np.arange(len(RESULT_CATEGORIES)) != RESULT_CATEGORIES.index(observations["results_view"]), :] = 0.
        if "xg_view" in observations:
            joint[:, :, np.arange(len(XG_CATEGORIES)) != XG_CATEGORIES.index(observations["xg_view"])] = 0.
        evidence_probability = float(joint.sum())
        if evidence_probability <= 0:
            raise ValueError("Impossible observed category under fitted contextual joint")
        joint /= evidence_probability
        return {"probabilities": joint.tolist(), "prior_probabilities": list(prior["probabilities"]),
                "current_probabilities": joint.sum(axis=(1, 2)).tolist() if observations else list(prior["probabilities"]),
                "observations": observations, "results_categories": list(RESULT_CATEGORIES),
                "xg_categories": list(XG_CATEGORIES), "evidence_probability": evidence_probability,
                "q_support": audit, "base_model_version": self.base_predictor.model_version,
                "model_version": self.model_version}

    def observation_support(self, state):
        # Population support is unchanged: local counts are partially pooled,
        # not a new isolated stratum. Report the additional support explicitly.
        support = super().observation_support(state)
        audit = self.joint_distribution(state)["q_support"]
        support.update(contextual_support={k: v for k, v in audit.items() if k.startswith("local_")},
                       scope="population conditional coverage with explicitly reported pooled local contribution")
        return support

    def _payload(self):
        return {"schema": VERSION, "reference": self._reference_payload,
                "fixture_contexts": self._fixture_rows,
                "context_weight": self.context_weight, "evidence_strength": self.evidence_strength,
                "selection": self._selection}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported contextual joint artifact schema")
        model = cls(_restore_reference(payload["reference"]), payload["fixture_contexts"],
                    context_weight=payload["context_weight"], strength=payload["evidence_strength"],
                    selection=payload["selection"])
        if model.model_version != payload.get("model_version"):
            raise ValueError("Contextual joint artifact hash mismatch")
        return model

    @classmethod
    def fit_context(cls, reference, fixture_rows, heldout_rows, actions, *, as_of, min_matches=12):
        """Past held-out selection; evaluation rows never enter this function.

        The three context candidates are fixed, not searched on evaluation
        performance. For each, optimize the one-dimensional convex log-score
        evidence-strength objective, then choose held-out loss; ties prefer
        population context. All three acquired sets count equally per match.
        """
        # Reuse existing chronology, identity, result, and target-disjointness
        # validation. Its zero-context result is also an exact baseline.
        baseline = ConservativeJointReturnPredictor.fit_strength(
            reference, heldout_rows, actions, as_of=as_of, min_matches=min_matches)
        validation = baseline.selection_report
        rows = sorted(heldout_rows, key=lambda r: (parse_time(r["prediction_cutoff"]), str(r["match_id"])))
        # Materialize only allow-listed reference fixture identities, never
        # training row feature/label payloads or excluded future rows.
        prototype = cls(reference, fixture_rows, context_weight=0., strength=1.)
        candidates, selected_samples = [], {}
        for weight in CONTEXT_WEIGHTS:
            model = cls(reference, prototype._fixture_rows, context_weight=weight, strength=1.)
            priors, posteriors, samples = [], [], []
            for row in rows:
                label = outcome(row)
                before = model.predict_partial(state_after(row, actions, []))["probabilities"]
                after = [model.predict_partial(state_after(row, actions, acquired))["probabilities"]
                         for acquired in ACQUIRED_SETS]
                priors.append([before[label]]*len(ACQUIRED_SETS))
                posteriors.append([p[label] for p in after])
                samples.append({"match_id": str(row["match_id"]), "outcome_index": label,
                                "initial_probabilities": before, "acquired_set_probabilities": after})
            optimization = minimize_strength(priors, posteriors)
            candidates.append({"context_weight": weight, **optimization,
                               "sample_digest": _digest(samples)})
            selected_samples[weight] = samples
        selected = min(candidates, key=lambda x: (x["selected_objective"], x["context_weight"]))
        # Guard against accidental divergence from the existing zero-context
        # objective, rather than silently changing the comparison predictor.
        if abs(candidates[0]["selected_objective"]-validation["optimization"]["selected_objective"]) > 1e-12:
            raise ValueError("Zero-context baseline does not reproduce existing strength selection")
        selection = {
            "schema": VERSION, "as_of": parse_time(as_of).isoformat(),
            "reference_payload_sha256": _digest(prototype._reference_payload),
            "fixture_context_sha256": _digest(prototype._fixture_rows),
            "dataset_version": validation["dataset_version"], "match_ids": validation["match_ids"],
            "reference_model_version": reference.model_version,
            "chronology_validation_sample_digest": validation["sample_digest"],
            "latest_label_available_at": validation["latest_label_available_at"],
            "minimum_matches": min_matches, "context_weight_candidates": list(CONTEXT_WEIGHTS),
            "population_prior_mass_per_outcome": reference.dirichlet_mass,
            "acquired_sets": [list(x) for x in ACQUIRED_SETS],
            "selection_target": "equal_match_equal_acquired_set_log_loss",
            "tie_break": "lowest_context_weight", "candidates": candidates, "selected": selected,
            "samples": selected_samples[selected["context_weight"]],
            "costs_or_query_thresholds_changed": False,
            "scope": "past held-out likelihood selection; no empirical VoI calibration guarantee",
        }
        return cls(reference, prototype._fixture_rows, context_weight=selected["context_weight"],
                   strength=selected["strength"], selection=selection)
