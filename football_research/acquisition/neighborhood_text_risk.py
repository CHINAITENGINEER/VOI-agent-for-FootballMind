"""Signed update-risk calibration on declared public ordered request contexts.

The predictor is the unchanged frozen R186 neighborhood model. Only critic
grouping is coarsened: actual text tokens and order remain in predictor state.
This is empirical residual pooling, not Bayes EVSI or a risk certificate.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from .expected_voi import CandidateValue, COMPUTED, INELIGIBLE
from .round9_residual import BIN_EDGES, evsi_bin, residual_summary
from .state import parse_time
from .text_history import HistoryNeighborhoodTextReturnModel, HISTORY_KERNEL, VERSION as TEXT_VERSION
from .text_returns import NEUTRAL, ORIGINS, _canonical, _digest
from .text_risk import TextRiskCalibrator, VERSION as LEGACY_VERSION, VALUE_KIND, _normalized, _number
from .value import _confidence

VERSION = "neighborhood-text-empirical-risk-v1"
CONFIDENCE = ("max_probability_lt_0.45", "max_probability_ge_0.45")
CONTEXT_SPEC = {
    "version": "ordered-public-request-spine-v1",
    "parent": ["candidate_channel", "candidate_request_cluster",
               "complete_ordered_channel_request_response_kind_spine", "prequery_support_origin"],
    "response_kind": "article_for_text_tokens_otherwise_original_typed_neutral_token",
    "preserve": ["actual_order", "multiplicity", "typed_neutral_events"],
    "support_origin": ["exact_only", "neighbor_only", "mixed"],
    "local": ["existing_confidence_max_probability_lt_0.45", "existing_model_evsi_bin"],
    "evsi_bin_edges": list(BIN_EDGES),
    "fallback": "same_parent_only_never_cross_source_request_order_or_support_origin",
    "excluded": ["exact_article_response_token", "unread_response", "future_menu", "label", "latent_family"],
    "assumption": "empirical_residual_pooling_not_conditional_sufficiency_or_stationarity_proof",
}
FORBIDDEN_SUPERVISION = {"family", "latent", "latent_family", "scenario_family", "stratum",
                         "expected_gain", "target_gain", "support_origin", "parent_context", "context_key"}


def _model_hash(model):
    if (not isinstance(model, HistoryNeighborhoodTextReturnModel) or model.version != TEXT_VERSION
            or not model.model_version or model.metadata.get("history_kernel") != HISTORY_KERNEL):
        raise ValueError("The frozen v2 history-neighborhood text predictor is required")
    content = _digest(model._payload())
    if model.model_version != TEXT_VERSION+":"+content[:24]:
        raise ValueError("Frozen neighborhood predictor content no longer matches its version")
    return content


def _support_origin(enumeration):
    if enumeration["status"] != "supported":
        return None
    support = enumeration["support"]
    exact, neighbor = support["history_exact_transition_count"], support["history_neighbor_transition_count"]
    if exact > 0 and neighbor > 0:
        return "mixed"
    if exact > 0:
        return "exact_only"
    if neighbor > 0:
        return "neighbor_only"
    raise ValueError("Supported neighborhood enumeration has no history support origin")


class NeighborhoodTextRiskCalibrator(TextRiskCalibrator):
    """Separate critic/artifact identity; the R183 implementation is unchanged."""

    def _public_context(self, state, action, enumeration):
        _, channel, token, _ = self.text_model._encode_action(action)
        origin = _support_origin(enumeration)
        if token is None or origin is None:
            return None, None, None, origin
        spine = [[entry["channel"], entry["action_token"],
                  entry["response_token"] if entry["response_token"] in NEUTRAL else "article"]
                 for entry in state["history"]]
        parent = _canonical([channel, token, spine, origin])
        return parent, _confidence(state["current_probabilities"]), evsi_bin(enumeration["value"]["gross_evsi"]), origin

    def fit(self, records, *, text_model, as_of, dataset_version, evidence_origin):
        content = _model_hash(text_model)
        if evidence_origin not in ORIGINS or evidence_origin != text_model.metadata["evidence_origin"]:
            raise ValueError("Calibration and fixed predictor require the same explicit evidence origin")
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError("An explicit dataset version is required")
        if isinstance(records, (list, tuple)) and any(
                isinstance(row, dict) and FORBIDDEN_SUPERVISION.intersection(row) for row in records):
            raise ValueError("Supply raw transitions, not hidden family or handed-in critic features")
        as_of = parse_time(as_of)
        raw = _normalized(records, text_model, as_of, evidence_origin)
        self.text_model = text_model
        derived, incomplete = [], set()
        for row in raw:
            state = text_model.start(row["prior_probabilities"], row["match_id"], row["prediction_cutoff"])
            for step in row["history"]:
                state = text_model.update(state, step["action"], step["response"])["state"]
            before = list(state["current_probabilities"])
            enumeration = text_model.enumerate(state, row["action"], cost=0., loss="log_loss")
            model_evsi = enumeration["value"]["gross_evsi"]
            parent, confidence, bucket, origin = self._public_context(state, row["action"], enumeration)
            update = text_model.update(state, row["action"], row["response"], decision=enumeration)
            supported = enumeration["status"] == "supported"
            eligible = supported and update["status"] == "updated"
            after = update["prediction_after"]
            gain = math.log(after[row["outcome"]])-math.log(before[row["outcome"]]) if eligible else None
            # Unsupported queries are outside the declared calibration
            # population. An unknown actual return after a supported query is
            # missing supervision within that population and blocks pooling.
            if supported and not eligible:
                incomplete.add(parent)
            derived.append({"match_id": row["match_id"], "prediction_cutoff": row["prediction_cutoff"],
                "result_available_at": row["result_available_at"], "raw_transition_sha256": _digest(row),
                "context_key": parent, "confidence": confidence, "evsi_bin": bucket,
                "support_origin": origin, "model_evsi": model_evsi,
                "prediction_before": before, "prediction_after": after, "gain": gain,
                "residual": gain-model_evsi if eligible else None, "prequery_supported": supported,
                "eligible": eligible, "reason": "replayed_fixed_predictor_update" if eligible else update["reason"],
                "response_update_status": update["status"], "observed_token": update["observed_token"]})
        parents = defaultdict(list)
        for row in derived:
            if row["eligible"]:
                parents[row["context_key"]].append(row)
        cells = {}
        for parent, rows in sorted(parents.items()):
            broad = self._match_means(rows)
            for confidence in CONFIDENCE:
                for bucket in range(len(BIN_EDGES)+1):
                    cells[_canonical([parent, confidence, bucket])] = residual_summary(self._match_means(
                        [row for row in rows if row["confidence"] == confidence and row["evsi_bin"] == bucket]),
                        broad, **self.parameters)
        self.raw_records, self.records, self.cells = raw, derived, cells
        self.incomplete_contexts = sorted(incomplete)
        self.manifest = {
            "training_as_of": as_of.isoformat(), "dataset_version": dataset_version.strip(),
            "training_match_ids": sorted({row["match_id"] for row in raw}), "training_records": len(raw),
            "eligible_records": sum(row["eligible"] for row in derived),
            "unmodeled_records": sum(not row["eligible"] for row in derived),
            "prequery_supported_records": sum(row["prequery_supported"] for row in derived),
            "prequery_unsupported_records": sum(not row["prequery_supported"] for row in derived),
            "supported_query_unknown_returns": sum(row["prequery_supported"] and not row["eligible"] for row in derived),
            "latest_label_available_at": max(row["result_available_at"] for row in raw),
            "text_model_version": text_model.model_version, "text_model_content_sha256": content,
            "text_model_training_as_of": text_model.metadata["training_as_of"],
            "evidence_origin": evidence_origin, "production_ready": False, "value_kind": VALUE_KIND,
            "operational_clock_eligible": evidence_origin == "real_observational",
            "label_contract": "independent_settlement_sidecar_trusted_from_verified_dataset_gateway_not_verified_by_outcome_integer",
            "target": "internally_replayed_log_loss_before_minus_after_minus_model_evsi",
            "supervision": "same_frozen_v2_predictor_heldout_actual_ordered_responses",
            "context_spec": deepcopy(CONTEXT_SPEC), "calibration_population": "prequery_supported_only",
            "support_unit": "one_mean_per_match_context_over_unique_observed_transitions",
            "time_blocks": "UTC_ISO_week_of_prediction_cutoff_in_legacy_kickoff_slot",
            "clock_scope": "supplied_capture_and_decision_clocks_checked_missing_clocks_not_invented",
            "unknown_policy": "prequery_unknown_audited_outside_population_supported_query_unknown_return_blocks_parent",
            "uncertainty_kind": "descriptive_week_cluster_mean_standard_error_no_coverage_guarantee",
            "selection_scope": "observed_supported_public_context_not_open_world_or_causal_identification",
            "predictor_parameter_uncertainty": "not_estimated", "risk_mode": "mean_only"}
        self.model_version = VERSION+":"+_digest(self._payload())[:24]
        return self

    def validate_state(self, state):
        if not self.model_version or self.text_model is None:
            raise ValueError("Fit or restore the neighborhood risk calibrator first")
        if (_model_hash(self.text_model) != self.manifest["text_model_content_sha256"] or
                self.text_model.model_version != self.manifest["text_model_version"]):
            raise ValueError("Risk calibrator is bound to a different frozen neighborhood predictor")
        clean = self.text_model._state(state)
        if (clean["match_id"] in self.manifest["training_match_ids"] or
                not parse_time(self.manifest["training_as_of"]) < parse_time(clean["prediction_cutoff"])):
            raise ValueError("Risk inference requires a new match strictly after calibration fitting")
        return clean

    def estimate(self, state, action, *, cost=0., loss="log_loss"):
        if loss != "log_loss":
            raise ValueError("Neighborhood text risk supports log_loss only")
        cost = _number(cost, "cost")
        state = self.validate_state(state)
        enumeration = self.text_model.enumerate(state, action, cost=cost, loss=loss)
        model_evsi = enumeration["value"]["gross_evsi"]
        parent, confidence, bucket, origin = self._public_context(state, action, enumeration)
        summary, reason = None, enumeration["reason"]
        if enumeration["status"] == "supported":
            summary = self.cells.get(_canonical([parent, confidence, bucket]))
            reason = ("incomplete_observed_response_coverage" if parent in self.incomplete_contexts else
                      "unsupported_empirical_context" if summary is None else
                      "insufficient_empirical_matches" if summary["residual_mean"] is None else
                      "computed_empirical_predictor_update_gain")
        computed = reason == "computed_empirical_predictor_update_gain"
        gain = model_evsi+summary["residual_mean"] if computed else None
        visible = {key: value for key, value in (summary or {}).items()
                   if key not in {"local_match_ids", "parent_match_ids", "week_influences"}}
        empirical = {**visible, "status": "computed" if computed else "unknown", "reason": reason,
            "expected_gain": gain, "model_evsi": model_evsi, "context_key": parent,
            "confidence": confidence, "evsi_bin": bucket, "support_origin": origin,
            "value_kind": VALUE_KIND, "estimator_version": self.model_version,
            "evidence_origin": self.manifest["evidence_origin"], "risk_mode": "mean",
            "uncertainty_kind": self.manifest["uncertainty_kind"], "stop_certificate": None,
            "production_ready": False}
        value = CandidateValue(candidate_id=enumeration["value"]["candidate_id"],
            status=COMPUTED if computed else INELIGIBLE, loss=loss, prior_risk=None, expected_after_risk=None,
            gross_evsi=None, cost=cost, delay_penalty=None, reliability_penalty=None,
            net_voi=gain-cost if computed else None, coherence_error=None, reason=reason,
            predictor_version=self.text_model.model_version, response_model_version=self.model_version,
            calibration_cutoff=self.manifest["training_as_of"])
        return {"value": value.as_dict(), "empirical_estimate": empirical, "model_enumeration": enumeration}

    def _payload(self):
        return {"schema": VERSION, "parameters": self.parameters, "manifest": self.manifest,
                "raw_records": self.raw_records, "records": self.records, "cells": self.cells,
                "incomplete_contexts": self.incomplete_contexts}

    def save(self, path):
        if not self.model_version or _model_hash(self.text_model) != self.manifest["text_model_content_sha256"]:
            raise ValueError("Only a fitted calibrator with its bound predictor can be saved")
        payload = self._payload()
        if self.model_version != VERSION+":"+_digest(payload)[:24]:
            raise ValueError("Neighborhood risk content no longer matches its fitted version")
        Path(path).write_text(json.dumps({**payload, "model_version": self.model_version,
            "content_sha256": _digest(payload)}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, text_model):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate neighborhood risk artifact field")
                result[key] = value
            return result
        def reject(value):
            raise ValueError("Nonfinite neighborhood risk artifact constant: "+value)
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)
        expected = {"schema", "parameters", "manifest", "raw_records", "records", "cells",
                    "incomplete_contexts", "model_version", "content_sha256"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != VERSION:
            raise ValueError("Unexpected frozen neighborhood text-risk schema")
        payload = {key: value for key, value in data.items() if key not in {"model_version", "content_sha256"}}
        if data["content_sha256"] != _digest(payload) or data["model_version"] != VERSION+":"+_digest(payload)[:24]:
            raise ValueError("Frozen neighborhood text-risk artifact hash mismatch")
        if (_model_hash(text_model) != data["manifest"]["text_model_content_sha256"] or
                text_model.model_version != data["manifest"]["text_model_version"]):
            raise ValueError("Neighborhood text-risk artifact requires its exact bound predictor")
        model = cls(**data["parameters"]).fit(data["raw_records"], text_model=text_model,
            as_of=data["manifest"]["training_as_of"], dataset_version=data["manifest"]["dataset_version"],
            evidence_origin=data["manifest"]["evidence_origin"])
        if model._payload() != payload or model.model_version != data["model_version"]:
            raise ValueError("Saved neighborhood statistics do not reproduce raw fixed-predictor replay")
        return model


def load_text_risk_model(path, *, text_model):
    """Explicit old/new schema dispatch; both loaders validate and replay raw data."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = {LEGACY_VERSION: TextRiskCalibrator, VERSION: NeighborhoodTextRiskCalibrator}.get(
        data.get("schema") if isinstance(data, dict) else None)
    if model_type is None:
        raise ValueError("Unsupported frozen text-risk model schema")
    return model_type.load(path, text_model=text_model)


__all__ = ["NeighborhoodTextRiskCalibrator", "load_text_risk_model", "VERSION", "CONTEXT_SPEC"]
