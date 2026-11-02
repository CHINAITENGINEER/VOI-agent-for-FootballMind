"""Opt-in local smoothing of ordered text acquisition histories.

Only one adjacent exchange is supported, subject to strict observed-token
gates. This is a testable conditional smoothing assumption, not a claim that
evidence order is irrelevant or that the response model is calibrated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from .text_returns import TextReturnModel, VERSION as EXACT_VERSION, NEUTRAL, _digest
from .state import parse_time

VERSION = "ordered-text-return-neighborhood-v2"
STATE_VERSION = "ordered-text-return-state-v2"
NEIGHBOR_WEIGHT = math.exp(-.5)
HISTORY_KERNEL = {
    "version": "single-adjacent-history-neighborhood-v1",
    "max_adjacent_swaps": 1,
    "swap_penalty": .5,
    "weight": "exp(-swap_penalty*adjacent_swap_count)",
    "gates": ["same_complete_channel_request_response_multiset",
              "same_response_order_within_each_channel_request_cluster",
              "same_ordered_typed_neutral_subsequence"],
    "unknown_history": "reject_entire_chain_before_neighbor_lookup",
    "history_retention": "complete_actual_order_in_state_and_training_history_key",
    "match_support": "sum_of_per_match_mean_history_weights_after_action_and_positive_prior_kernel_filter",
    "minimum_history_mass": "existing_min_matches_not_normalized_effective_sample_size",
    "minimum_history_mean": NEIGHBOR_WEIGHT,
    "interpretation": "conditional_smoothing_hypothesis_not_exchangeability_or_independence_proof",
}


def _triples(history):
    if (not isinstance(history, list) or len(history) > 32
            or any(not isinstance(entry, list) or len(entry) != 3
                   or any(not isinstance(value, str) or not value for value in entry) for entry in history)):
        raise ValueError("A complete ordered list of encoded history triples is required")
    return tuple(tuple(entry) for entry in history)


def _request_sequences(history):
    grouped = defaultdict(list)
    for channel, request, response in history:
        grouped[(channel, request)].append(response)
    return {key: tuple(values) for key, values in grouped.items()}


def history_swap_distance(left, right):
    """Return 0/1 for a permitted neighbor, None for any unsupported relation."""
    left, right = _triples(left), _triples(right)
    if (Counter(left) != Counter(right) or _request_sequences(left) != _request_sequences(right)
            or tuple(x for x in left if x[2] in NEUTRAL) != tuple(x for x in right if x[2] in NEUTRAL)):
        return None
    if left == right:
        return 0
    for index in range(len(left)-1):
        swapped = left[:index]+(left[index+1], left[index])+left[index+2:]
        if swapped == right:
            return 1
    return None


def _check_supplied_clocks(records):
    """Missing clocks remain missing; explicitly impossible histories cannot train."""
    for record in records:
        cutoff = parse_time(record["prediction_cutoff"])
        previous_available = None
        history = record.get("history", [])
        if not isinstance(history, list) or len(history) > 32:
            raise ValueError("A finite ordered observed history is required")
        for step in history+[record]:
            clocks = {}
            for key in ("decision_at", "observed_at", "available_at"):
                values = [parse_time(source[key]) for source in (step, step.get("response", {}))
                          if source.get(key) is not None]
                if any(value > cutoff for value in values):
                    raise ValueError("Provided history/current clock exceeds prediction cutoff")
                if len(set(values)) > 1:
                    raise ValueError("Conflicting history/current clocks")
                if values:
                    clocks[key] = values[0]
            known = [clocks[key] for key in ("decision_at", "observed_at", "available_at") if key in clocks]
            if known != sorted(known):
                raise ValueError("Provided history/current clocks are out of order")
            if previous_available is not None and "decision_at" in clocks and previous_available > clocks["decision_at"]:
                raise ValueError("History was not available before the following decision")
            previous_available = clocks.get("available_at")


class HistoryNeighborhoodTextReturnModel(TextReturnModel):
    version = VERSION
    state_version = STATE_VERSION

    def fit(self, records, **kwargs):
        if isinstance(records, (list, tuple)):
            _check_supplied_clocks(records)
        super().fit(records, **kwargs)
        self.metadata.update(history_kernel=deepcopy(HISTORY_KERNEL),
            history_scope="complete_ordered_history_with_one_adjacent_exchange_conditional_smoothing",
            support_scope="unsmoothed_match_capped_support_and_absolute_history_weight_mass",
            order_invariance_established=False,
            value_scope="experimental_adjacent_history_conditional_text_response_model")
        self.model_version = self.version+":"+_digest(self._payload())[:24]
        return self

    def _history_neighbors(self, channel, key):
        ordered = _triples(key)
        result = super()._history_neighbors(channel, key)
        seen = {ordered}
        for index in range(len(ordered)-1):
            swapped = ordered[:index]+(ordered[index+1], ordered[index])+ordered[index+2:]
            if swapped in seen:
                continue
            seen.add(swapped)
            candidate = [list(entry) for entry in swapped]
            if history_swap_distance(key, candidate) == 1:
                result.extend((row, NEIGHBOR_WEIGHT) for row, _ in super()._history_neighbors(channel, candidate))
        return result

    def _history_support(self, observations):
        # The ordinary ESS is scale-invariant: a uniformly downweighted set
        # would otherwise retain its full support count. Preserve absolute
        # history mass without counting repeated transitions as new matches.
        per_match = defaultdict(list)
        for match_id, weight in observations:
            per_match[match_id].append(weight)
        means = [math.fsum(values)/len(values) for values in per_match.values()]
        mass = math.fsum(means)
        mean = mass/len(means) if means else 0.
        support = {"history_weighted_match_mass": mass, "history_mean_match_weight": mean,
            "history_weighted_support_minimum": self.parameters["min_matches"],
            "history_neighbor_weight": NEIGHBOR_WEIGHT,
            "history_exact_transition_count": sum(weight == 1. for _, weight in observations),
            "history_neighbor_transition_count": sum(weight != 1. for _, weight in observations),
            "history_mass_counts_as_independent_matches": False}
        reason = ("insufficient_history_neighbor_support" if
            mass+1e-9 < self.parameters["min_matches"] or mean+1e-12 < NEIGHBOR_WEIGHT else None)
        return support, reason

    def save(self, path):
        if (self.metadata.get("history_kernel") != HISTORY_KERNEL or
                self.model_version != self.version+":"+_digest(self._payload())[:24]):
            raise ValueError("History-neighborhood content or fixed specification changed")
        super().save(path)

    @classmethod
    def load(cls, path):
        model = super().load(path)
        if (model.metadata.get("history_kernel") != HISTORY_KERNEL
                or model.metadata.get("order_invariance_established") is not False):
            raise ValueError("Unsupported frozen history-neighborhood specification")
        parameters = model.parameters
        expected = {"bandwidth", "dirichlet_mass", "min_matches", "min_class_matches",
                    "text_clusters", "action_clusters", "min_action_similarity"}
        if not isinstance(parameters, dict) or set(parameters) != expected:
            raise ValueError("Invalid frozen history-neighborhood parameters")
        for key in ("bandwidth", "dirichlet_mass", "min_matches", "min_class_matches", "min_action_similarity"):
            value = parameters[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite history-neighborhood parameters required")
        if (parameters["min_action_similarity"] > 1 or any(type(parameters[key]) is not int or
                not 1 <= parameters[key] <= 32 for key in ("text_clusters", "action_clusters"))):
            raise ValueError("Invalid history-neighborhood cluster/similarity parameters")
        cutoff = parse_time(model.metadata["training_as_of"])
        outcomes, seen = {}, set()
        for row in model.records:
            _triples(row["history_key"])
            match_id, outcome = row["match_id"], row["outcome"]
            if (not isinstance(match_id, str) or not match_id.strip() or type(outcome) is not int or
                    outcome not in (0, 1, 2) or (match_id in outcomes and outcomes[match_id] != outcome)):
                raise ValueError("Inconsistent frozen history-neighborhood match outcomes")
            if not parse_time(row["prediction_cutoff"]) < parse_time(row["result_available_at"]) <= cutoff:
                raise ValueError("Frozen history-neighborhood settlement/training cutoff mismatch")
            identity = _digest(row)
            if identity in seen:
                raise ValueError("Duplicate frozen history-neighborhood transition")
            seen.add(identity)
            outcomes[match_id] = outcome
        if (not outcomes or sorted(outcomes) != model.metadata["training_match_ids"] or
                len(outcomes) != model.metadata["training_matches"] or
                len(model.records) != model.metadata["training_records"] or
                parse_time(model.metadata["trained_as_of"]) != cutoff):
            raise ValueError("Frozen history-neighborhood training metadata mismatch")
        return model


def load_text_model(path: str | Path) -> TextReturnModel:
    """Dispatch only registered artifact schemas; never guess or import a class."""
    schema = json.loads(Path(path).read_text(encoding="utf-8")).get("schema")
    model_type = {EXACT_VERSION: TextReturnModel, VERSION: HistoryNeighborhoodTextReturnModel}.get(schema)
    if model_type is None:
        raise ValueError("Unsupported frozen text-response model schema")
    # Both concrete loaders independently validate duplicate keys, finite
    # values, full artifact hash, version, and their own fixed schema.
    return model_type.load(path)


__all__ = ["HistoryNeighborhoodTextReturnModel", "load_text_model", "history_swap_distance", "VERSION", "STATE_VERSION"]
