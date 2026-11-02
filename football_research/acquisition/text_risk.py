"""Held-out empirical update-risk calibration for one frozen text predictor.

Raw observed transitions are replayed through the fixed predictor; supplied
loss improvements are never trusted. This reuses the existing match-weighted
residual summary. It is a calibration bridge, not a new Bayes VoI estimator.
Unknown returns are retained as missing support, never assigned a zero gain.
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
from .text_returns import TextReturnModel, VERSION as TEXT_VERSION, ORIGINS, _action, _response, _p, _canonical, _digest
from .value import _confidence

VERSION = "frozen-text-empirical-risk-v1"
VALUE_KIND = "empirical_predictor_risk_reduction_not_bayes_evsi"
CLOCKS = ("decision_at", "observed_at", "available_at", "source_published_at")
DERIVED = {"delta", "gain", "residual", "model_evsi", "prediction_before", "prediction_after"}


def _model_hash(model):
    if not isinstance(model, TextReturnModel) or not model.model_version:
        raise ValueError("A fitted frozen TextReturnModel is required")
    content = _digest(model._payload())
    if model.model_version != TEXT_VERSION + ":" + content[:24]:
        raise ValueError("Frozen text model content no longer matches its version")
    return content


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def _step(action, response, fields, *, cutoff, model_at):
    clean = _action(action)[0]
    _response(response)
    observed = {key: deepcopy(response[key]) for key in ("status", "kind", "text") if key in response}
    clocks = {}
    for key in CLOCKS:
        stamps = [parse_time(source[key]) for source in (fields, response) if source.get(key) is not None]
        if stamps:
            if len(set(stamps)) != 1 or stamps[0] > cutoff:
                raise ValueError("Observed transition has conflicting or post-cutoff clocks")
            clocks[key] = stamps[0]
    ordered = [clocks[key] for key in ("decision_at", "observed_at", "available_at") if key in clocks]
    if ordered != sorted(ordered):
        raise ValueError("Response clocks must follow decision, observation and availability order")
    # With no request clock, do not claim the capture was contemporaneous.
    first = clocks.get("decision_at", clocks.get("observed_at", clocks.get("available_at")))
    if first is not None and not model_at < first:
        raise ValueError("The fixed text model must predate the actual observed decision")
    if "source_published_at" in clocks and "observed_at" in clocks and clocks["source_published_at"] > clocks["observed_at"]:
        raise ValueError("Publication cannot follow the supplied capture time")
    return {"action": clean, "response": observed,
            **{key: stamp.isoformat() for key, stamp in clocks.items()}}


def _normalized(records, text_model, as_of, evidence_origin):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Nonempty raw held-out response records are required")
    seen, facts, prefixes = {}, {}, {}
    trained = set(text_model.metadata["training_match_ids"])
    model_at = parse_time(text_model.metadata["training_as_of"])
    for row in records:
        if not isinstance(row, dict) or DERIVED.intersection(row):
            raise ValueError("Supply raw transitions, not externally computed gain or predictions")
        mid, outcome = row.get("match_id"), row.get("outcome")
        if not isinstance(mid, str) or not mid.strip() or type(outcome) is not int or outcome not in (0, 1, 2):
            raise ValueError("An explicit match and independently settled outcome are required")
        cutoff, result_at = parse_time(row["prediction_cutoff"]), parse_time(row["result_available_at"])
        if mid in trained or not model_at < cutoff < result_at <= as_of:
            raise ValueError("Risk calibration requires held-out matches and strict frozen-model/label chronology")
        kickoff = parse_time(row["kickoff"]) if row.get("kickoff") is not None else None
        if kickoff is not None and not cutoff < kickoff <= result_at:
            raise ValueError("Known kickoff must lie between prediction and independent settlement")
        p0 = _p(row["prior_probabilities"])
        fact = (outcome, tuple(p0), cutoff, result_at, kickoff)
        if mid in facts and facts[mid] != fact:
            raise ValueError("Conflicting label, initial probabilities or chronology for one match")
        facts[mid] = fact
        history = row.get("history", [])
        if not isinstance(history, list) or len(history) > 32:
            raise ValueError("A bounded ordered actual response history is required")
        steps, prefix, latest = [], [], None
        for raw in history + [{"action": row["action"], "response": row["response"],
                              **{key: row[key] for key in CLOCKS if row.get(key) is not None}}]:
            step = _step(raw["action"], raw["response"], raw, cutoff=cutoff, model_at=model_at)
            if evidence_origin == "real_observational" and not all(
                    key in step for key in ("decision_at", "observed_at", "available_at")):
                raise ValueError("Real observations require decision, observation and availability clocks for every step")
            decision = parse_time(step["decision_at"]) if "decision_at" in step else None
            if decision is not None and latest is not None and decision < latest:
                raise ValueError("A later decision precedes the previously acquired response")
            stamps = [parse_time(step[key]) for key in ("decision_at", "observed_at", "available_at") if key in step]
            if stamps:
                if latest is not None and max(stamps) < latest:
                    raise ValueError("Observed acquisition history moves backwards in time")
                latest = max(stamps)
            identity = _canonical([mid, prefix, step["action"]])
            if identity in prefixes and prefixes[identity] != step:
                raise ValueError("Conflicting response or clocks for the same actual transition")
            prefixes[identity] = step
            prefix.append({"action": step["action"], "response": step["response"]})
            steps.append(step)
        record = {"match_id": mid, "outcome": outcome, "prior_probabilities": p0,
            "prediction_cutoff": cutoff.isoformat(), "result_available_at": result_at.isoformat(),
            "history": steps[:-1], **steps[-1]}
        if kickoff is not None:
            record["kickoff"] = kickoff.isoformat()
        if identity in seen and seen[identity] != record:
            raise ValueError("Conflicting duplicate actual transition")
        seen[identity] = record
    return [seen[key] for key in sorted(seen)]


class TextRiskCalibrator:
    """Mean signed update gain on the supported observed acquisition population."""

    def __init__(self, *, min_support=12, shrinkage=20.):
        if type(min_support) is not int or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        self.parameters = {"min_support": min_support, "shrinkage": _number(shrinkage, "shrinkage")}
        self.text_model, self.manifest, self.model_version = None, {}, ""
        self.raw_records, self.records, self.cells, self.incomplete_contexts = [], [], {}, []

    def _context(self, state, action, model_evsi):
        _, channel, token, _ = self.text_model._encode_action(action)
        if token is None:
            return None, None
        broad = _canonical([channel, token, self.text_model._history_key(state["history"]),
                            _confidence(state["current_probabilities"])])
        return broad, evsi_bin(model_evsi) if model_evsi is not None else None

    @staticmethod
    def _match_means(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["match_id"]].append(row)
        # The legacy residual_summary kickoff slot is a UTC week anchor only.
        # This adapter explicitly uses prediction cutoff, never invents kickoff.
        return [{"match_id": mid, "kickoff": group[0]["prediction_cutoff"],
                 "residual": math.fsum(row["residual"] for row in group) / len(group)}
                for mid, group in sorted(groups.items())]

    def fit(self, records, *, text_model, as_of, dataset_version, evidence_origin):
        content = _model_hash(text_model)
        if evidence_origin not in ORIGINS or evidence_origin != text_model.metadata["evidence_origin"]:
            raise ValueError("Calibration and fixed predictor require the same explicit evidence origin")
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError("An explicit dataset version is required")
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
            context, bucket = self._context(state, row["action"], model_evsi)
            update = text_model.update(state, row["action"], row["response"], decision=enumeration)
            eligible = enumeration["status"] == "supported" and update["status"] == "updated"
            after = update["prediction_after"]
            gain = math.log(after[row["outcome"]]) - math.log(before[row["outcome"]]) if eligible else None
            if context is not None and not eligible:
                incomplete.add(context)
            derived.append({"match_id": row["match_id"], "prediction_cutoff": row["prediction_cutoff"],
                "result_available_at": row["result_available_at"], "raw_transition_sha256": _digest(row),
                "context_key": context, "evsi_bin": bucket, "model_evsi": model_evsi,
                "prediction_before": before, "prediction_after": after, "gain": gain,
                "residual": gain - model_evsi if eligible else None,
                "eligible": eligible, "reason": "replayed_fixed_predictor_update" if eligible else update["reason"],
                "response_update_status": update["status"], "observed_token": update["observed_token"]})
        parents = defaultdict(list)
        for row in derived:
            if row["eligible"]:
                parents[row["context_key"]].append(row)
        cells = {}
        for context, rows in sorted(parents.items()):
            broad = self._match_means(rows)
            for bucket in range(len(BIN_EDGES) + 1):
                cells[_canonical([context, bucket])] = residual_summary(
                    self._match_means([row for row in rows if row["evsi_bin"] == bucket]), broad, **self.parameters)
        self.raw_records, self.records, self.cells = raw, derived, cells
        self.incomplete_contexts = sorted(incomplete)
        self.manifest = {"training_as_of": as_of.isoformat(), "dataset_version": dataset_version.strip(),
            "training_match_ids": sorted({row["match_id"] for row in raw}), "training_records": len(raw),
            "eligible_records": sum(row["eligible"] for row in derived),
            "unmodeled_records": sum(not row["eligible"] for row in derived),
            "latest_label_available_at": max(row["result_available_at"] for row in raw),
            "text_model_version": text_model.model_version, "text_model_content_sha256": content,
            "text_model_training_as_of": text_model.metadata["training_as_of"],
            "evidence_origin": evidence_origin, "production_ready": False, "value_kind": VALUE_KIND,
            "operational_clock_eligible": evidence_origin == "real_observational",
            "label_contract": "independent_settlement_sidecar_trusted_from_verified_dataset_gateway_not_verified_by_outcome_integer",
            "target": "internally_replayed_log_loss_before_minus_after_minus_model_evsi",
            "supervision": "same_frozen_predictor_heldout_actual_ordered_responses",
            "context": "action_channel_request_cluster_ordered_observed_tokens_current_confidence",
            "confidence_rule": "existing_max_probability_lt_0.45", "evsi_bin_edges": list(BIN_EDGES),
            "support_unit": "one_mean_per_match_context_over_unique_observed_transitions",
            "time_blocks": "UTC_ISO_week_of_prediction_cutoff_in_legacy_kickoff_slot",
            "clock_scope": "supplied_capture_and_decision_clocks_checked_missing_clocks_not_invented",
            "unknown_policy": "unmodeled_returns_retained_and_their_prequery_context_not_estimated",
            "uncertainty_kind": "descriptive_week_cluster_mean_standard_error_no_coverage_guarantee",
            "selection_scope": "observed_supported_context_not_open_world_or_causal_identification",
            "predictor_parameter_uncertainty": "not_estimated", "risk_mode": "mean_only"}
        self.model_version = VERSION + ":" + _digest(self._payload())[:24]
        return self

    def validate_state(self, state):
        if not self.model_version or self.text_model is None:
            raise ValueError("Fit or restore the text risk calibrator first")
        if (_model_hash(self.text_model) != self.manifest["text_model_content_sha256"]
                or self.text_model.model_version != self.manifest["text_model_version"]):
            raise ValueError("Risk calibrator is bound to a different frozen text predictor")
        clean = self.text_model._state(state)
        if (clean["match_id"] in self.manifest["training_match_ids"]
                or not parse_time(self.manifest["training_as_of"]) < parse_time(clean["prediction_cutoff"])):
            raise ValueError("Risk inference requires a new match strictly after calibration fitting")
        return clean

    def estimate(self, state, action, *, cost=0., loss="log_loss"):
        if loss != "log_loss":
            raise ValueError("Text risk calibration currently supports log_loss only")
        cost = _number(cost, "cost")
        state = self.validate_state(state)
        enumeration = self.text_model.enumerate(state, action, cost=cost, loss=loss)
        model_evsi = enumeration["value"]["gross_evsi"]
        context, bucket = self._context(state, action, model_evsi)
        summary, reason = None, enumeration["reason"]
        if enumeration["status"] == "supported":
            summary = self.cells.get(_canonical([context, bucket]))
            reason = ("incomplete_observed_response_coverage" if context in self.incomplete_contexts else
                      "unsupported_empirical_context" if summary is None else
                      "insufficient_empirical_matches" if summary["residual_mean"] is None else
                      "computed_empirical_predictor_update_gain")
        computed = reason == "computed_empirical_predictor_update_gain"
        gain = model_evsi + summary["residual_mean"] if computed else None
        visible_summary = {k: v for k, v in (summary or {}).items()
                           if k not in {"local_match_ids", "parent_match_ids", "week_influences"}}
        empirical = {**visible_summary, "status": "computed" if computed else "unknown", "reason": reason,
            "expected_gain": gain, "model_evsi": model_evsi, "context_key": context, "evsi_bin": bucket,
            "value_kind": VALUE_KIND, "estimator_version": self.model_version, "evidence_origin": self.manifest["evidence_origin"],
            "risk_mode": "mean", "uncertainty_kind": self.manifest["uncertainty_kind"],
            "stop_certificate": None, "production_ready": False}
        value = CandidateValue(candidate_id=enumeration["value"]["candidate_id"],
            status=COMPUTED if computed else INELIGIBLE, loss=loss, prior_risk=None, expected_after_risk=None,
            gross_evsi=None, cost=cost, delay_penalty=None, reliability_penalty=None,
            net_voi=gain - cost if computed else None, coherence_error=None, reason=reason,
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
        if self.model_version != VERSION + ":" + _digest(payload)[:24]:
            raise ValueError("Risk calibrator content no longer matches its fitted version")
        Path(path).write_text(json.dumps({**payload, "model_version": self.model_version,
            "content_sha256": _digest(payload)}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, text_model):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate risk artifact field")
                result[key] = value
            return result
        def reject(value):
            raise ValueError("Nonfinite risk artifact constant: " + value)
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)
        expected = {"schema", "parameters", "manifest", "raw_records", "records", "cells",
                    "incomplete_contexts", "model_version", "content_sha256"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != VERSION:
            raise ValueError("Unexpected frozen text-risk schema")
        payload = {k: v for k, v in data.items() if k not in {"model_version", "content_sha256"}}
        if data["content_sha256"] != _digest(payload) or data["model_version"] != VERSION + ":" + _digest(payload)[:24]:
            raise ValueError("Frozen text-risk artifact hash mismatch")
        if (_model_hash(text_model) != data["manifest"]["text_model_content_sha256"]
                or text_model.model_version != data["manifest"]["text_model_version"]):
            raise ValueError("Text-risk artifact requires its exact bound predictor")
        # A rehashed supplied statistic is still not evidence: recompute from
        # the saved raw transitions with the same frozen f, without fitting f.
        model = cls(**data["parameters"]).fit(data["raw_records"], text_model=text_model,
            as_of=data["manifest"]["training_as_of"], dataset_version=data["manifest"]["dataset_version"],
            evidence_origin=data["manifest"]["evidence_origin"])
        if model._payload() != payload or model.model_version != data["model_version"]:
            raise ValueError("Saved empirical statistics do not reproduce raw fixed-predictor replay")
        return model


__all__ = ["TextRiskCalibrator", "VERSION", "VALUE_KIND"]
