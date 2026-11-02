"""Explicit learning exploration among generated, caller-approved candidates.

This module estimates no information value. The caller must exclude actions
with known zero or negative net value and freeze the configuration before the
run starts. Private seeded draws are reproducible design randomization, not a
claim about coverage of the open information environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import re


VERSION = "inquiry-learning-exploration-v1"
DESIGN = "uniform-generated-eligible-candidates"
SCOPE = "conditional_on_generated_eligible_candidates_only"
CONFIGURATION_KEY = "inquiry_exploration"
_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,100}\Z")
_KINDS = {"query", "stop", "wait", "defer", "reflect"}


def _identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty protocol identifier")
    return value


@dataclass(frozen=True)
class UniformInquiryExploration:
    """One reproducible, conditional uniform draw per pre-query decision index.

    ``configuration()`` is private run configuration. Only ``public_summary()``
    belongs in provider input. ``select`` reads no return, document, label or
    value field; eligibility is an explicit responsibility of the caller.
    """

    seed: int = field(repr=False)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**256:
            raise ValueError("Exploration seed must be an integer in [0, 2**256)")

    @property
    def seed_commitment(self):
        return hashlib.sha256(
            (VERSION + ":seed:").encode("ascii") + self.seed.to_bytes(32, "big")
        ).hexdigest()

    def configuration(self):
        return {"version": VERSION, "design": DESIGN, "scope": SCOPE,
                "seed": self.seed, "seed_commitment": self.seed_commitment}

    @classmethod
    def from_configuration(cls, configuration):
        required = {"version", "design", "scope", "seed", "seed_commitment"}
        if not isinstance(configuration, dict) or set(configuration) != required:
            raise ValueError("A complete frozen exploration configuration is required")
        instance = cls(configuration["seed"])
        if configuration != instance.configuration():
            raise ValueError("Unsupported exploration design or seed commitment mismatch")
        return instance

    def public_summary(self):
        return {"version": VERSION, "design": DESIGN, "scope": SCOPE,
                "seed_commitment": self.seed_commitment,
                "purpose": "bounded_learning_exploration_not_estimated_voi",
                "candidate_generation_probability": None,
                "counterfactual_utility": None,
                "seed_status": "private_precommitted_replayable_state"}

    def _draw_index(self, run_id, decision_index, count):
        # Counter-mode HMAC gives a domain-separated private pseudorandom stream.
        # Rejection sampling removes modulo bias when the range is not a power
        # of two. No mutable global RNG or response-dependent input is involved.
        limit = 2**256 - (2**256 % count)
        counter = 0
        while True:
            message = json.dumps(
                {"version": VERSION, "run_id": run_id,
                 "decision_index": decision_index, "counter": counter},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii")
            draw = int.from_bytes(hmac.new(self.seed.to_bytes(32, "big"), message,
                                          hashlib.sha256).digest(), "big")
            if draw < limit:
                return draw % count
            counter += 1

    def select(self, run, response, eligible_ids):
        """Preserve non-query decisions or draw among an explicit eligible subset.

        Call before appending the current decision to ``run['decisions']``.
        Its zero-based index is the number of previous decisions, including
        previous reflections. The frozen seed configuration must already be
        present at ``run['configuration']['inquiry_exploration']``.
        """
        if not isinstance(response, dict):
            raise ValueError("An existing inquiry response is required")
        decision = response.get("decision")
        if not isinstance(decision, dict) or decision.get("kind") not in _KINDS:
            raise ValueError("An existing inquiry decision is required")
        if decision["kind"] != "query":
            return {**deepcopy(decision), "exploration_record": None}

        if not isinstance(run, dict) or not isinstance(run.get("configuration"), dict):
            raise ValueError("Exploration configuration must match the frozen run configuration")
        frozen = run["configuration"].get(CONFIGURATION_KEY)
        if (not isinstance(frozen, dict) or type(frozen.get("seed")) is not int
                or frozen != self.configuration()):
            raise ValueError("Exploration configuration must match the frozen run configuration")
        run_id = _identifier(run.get("run_id"), "Run ID")
        prior_decisions = run.get("decisions")
        if not isinstance(prior_decisions, list):
            raise ValueError("Pre-query decision history must be a list")
        decision_index = len(prior_decisions)

        proposed = response.get("candidates")
        if not isinstance(proposed, list) or not proposed:
            raise ValueError("Query exploration requires generated candidates")
        proposed_ids = []
        for candidate in proposed:
            if not isinstance(candidate, dict):
                raise ValueError("Each generated candidate must have a protocol identifier")
            proposed_ids.append(_identifier(candidate.get("candidate_id"), "Candidate ID"))
        if len(proposed_ids) != len(set(proposed_ids)):
            raise ValueError("Generated candidate identifiers must be unique")
        if decision.get("candidate_id") not in proposed_ids or decision.get("revisit_at") is not None:
            raise ValueError("The original query must name a generated candidate with no revisit time")
        if not isinstance(eligible_ids, (list, tuple)) or not eligible_ids:
            raise ValueError("A nonempty explicit eligible candidate list is required")
        eligible = [_identifier(value, "Eligible candidate ID") for value in eligible_ids]
        if len(eligible) != len(set(eligible)):
            raise ValueError("Eligible candidate identifiers must be unique")
        if not set(eligible) <= set(proposed_ids):
            raise ValueError("Eligible candidates must be a subset of generated candidates")
        eligible.sort()
        chosen_id = eligible[self._draw_index(run_id, decision_index, len(eligible))]
        probability = 1 / len(eligible)
        record = {"version": VERSION, "design": DESIGN, "scope": SCOPE,
                  "run_id": run_id, "decision_index": decision_index,
                  "seed_commitment": self.seed_commitment,
                  "eligible_ids": eligible,
                  "probabilities": {candidate_id: probability for candidate_id in eligible},
                  "selected_p": probability, "chosen_id": chosen_id,
                  "original_candidate_id": decision["candidate_id"],
                  "eligibility_basis": "explicit_caller_subset_excluding_known_nonpositive_net_value",
                  "probability_basis": "precommitted_uniform_sampling_design",
                  "replay_scope": "deterministic_given_private_seed_run_and_decision_index",
                  "candidate_generation_probability": None,
                  "counterfactual_utility": None,
                  "expected_information_value": None}
        return {**deepcopy(decision), "candidate_id": chosen_id,
                "reason": "Learning exploration: uniformly selected from the explicitly eligible generated candidates; information value remains unestimated.",
                "exploration_record": record}
