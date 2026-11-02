"""State-erased historical means of fixed-predictor query gains.

These controls estimate empirical predictor-update gains, not model EVSI. The
only numerical keys are action identity, optionally the acquired-action set.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from .joint_returns import ACTION_IDS, HYPOTHESIS_KEY, JointReturnPredictor
from .joint_planner import _budget, _cost, _number, _probabilities, UNITS
from .round9_residual import context_key
from .state import parse_time, sanitize_state
from .value import _canonical

VERSION = "historical-gain-prior-v1"
MODES = ("action_only", "acquired_set")
ACTION_GROUPS = {"results_view": ("elo", "recent_form"), "xg_view": ("xg",)}
FAILURES = ("FAILED", "TIMEOUT", "PROVIDER_ERROR", "HTTP_ERROR", "NETWORK_ERROR")


def visible_context(state, *, model, trained_as_of=None):
    """Validate public boundaries, without predicting or extracting value features."""
    if HYPOTHESIS_KEY in state:
        raise ValueError("Hypothetical return tokens are forbidden at runtime")
    history = state.get("action_history", [])
    if not isinstance(history, (list, tuple)):
        raise ValueError("Action history must be a sequence")
    attempted = set()
    for event in history:
        if not isinstance(event, dict) or event.get("action_id") not in ACTION_IDS:
            raise ValueError("History requires a modeled action identity")
        if any(t in str(event.get("status", "")).upper() for t in FAILURES):
            raise ValueError("Provider failure is unmodeled, not missing evidence")
        attempted.add(event["action_id"])
    for item in state.get("groups", {}).values():
        if isinstance(item, dict) and item.get("status", "not_acquired") != "not_acquired":
            if any(t in (str(item.get("status", ""))+" "+str(item.get("reason", ""))).upper() for t in FAILURES):
                raise ValueError("Provider failure is unmodeled, not missing evidence")
    clean = sanitize_state(state)
    at = parse_time(clean["prediction_cutoff"])
    if (clean["information_mode"] != model.metadata["information_mode"]
            or at < parse_time(model.metadata["training_as_of"])
            or (trained_as_of is not None and at < parse_time(trained_as_of))):
        raise ValueError("Inference information mode/time boundary violation")
    acquired = []
    for aid, groups in ACTION_GROUPS.items():
        known = [clean["groups"][g]["status"] in {"observed", "source_missing"} for g in groups]
        if any(known) and not all(known):
            raise ValueError("Complete bundled tool projection required")
        if all(known):
            acquired.append(aid)
    return clean, sorted(acquired), attempted


def gain_summary(rows, *, min_support):
    """Rows already contain exactly one possibly averaged gain per match."""
    members = {}
    for r in rows:
        mid = r["match_id"]
        if mid in members:
            raise ValueError("Duplicate match in gain-prior cell")
        members[mid] = {"gain": float(r["gain"]), "kickoff": parse_time(r["kickoff"]).isoformat()}
    n = len(members)
    mean = math.fsum(r["gain"] for r in members.values())/n if n else None
    influences = defaultdict(float)
    for r in members.values():
        year, week, _ = parse_time(r["kickoff"]).isocalendar()
        influences[f"{year}-W{week:02d}"] += (r["gain"]-mean)/n
    blocks = len(influences)
    se = math.sqrt(blocks/(blocks-1)*math.fsum(v*v for v in influences.values())) if blocks >= 2 else None
    supported = n >= min_support
    return {"support": n, "support_matches": n, "parent_support": n, "effective_support": float(n),
        "time_blocks": blocks, "sample_mean": mean, "expected_gain": mean if supported else None,
        "standard_error": se, "support_status": "supported_local" if supported else "unsupported",
        "fallback_reason": None if supported else "unique_match_support_below_minimum",
        "match_values": dict(sorted(members.items())), "week_influences": dict(sorted(influences.items()))}


class GainPriorEstimator:
    def __init__(self, *, mode, min_support=12):
        if mode not in MODES or type(min_support) is not int or min_support < 1:
            raise ValueError("Declared erasure mode and positive integer support required")
        self.mode, self.min_support = mode, min_support
        self.records, self.cells, self.manifest = [], {}, {}
        self.model, self.version = None, VERSION

    def _key(self, action_id, acquired):
        context_key(action_id, acquired)  # Checks legal one-step context in both modes.
        return _canonical([action_id, sorted(acquired) if self.mode == "acquired_set" else None])

    def fit(self, samples, *, as_of, dataset_version, joint_model, joint_artifact_sha256, metadata=None):
        cutoff = parse_time(as_of)
        records, matches = {}, defaultdict(set)
        for sample in samples:
            clean, acquired, _ = visible_context(sample["state"], model=joint_model)
            mid, aid = str(sample["match_id"]), sample["action"]["action_id"]
            key = context_key(aid, acquired)
            if mid != clean["match_id"] or sorted(sample["acquired_action_ids"]) != acquired:
                raise ValueError("Supervision identity/acquired context mismatch")
            if (sample.get("supervision_kind") != "same_version_heldout_fixed_predictor"
                    or sample["oof_predictor_version"] != joint_model.model_version
                    or sample["oof_manifest_ref"] != "sha256:"+joint_artifact_sha256
                    or parse_time(sample["oof_train_cutoff"]) != parse_time(joint_model.metadata["training_as_of"])):
                raise ValueError("Same-version generating predictor binding required")
            pred, kick, label = map(parse_time, (clean["prediction_cutoff"], sample["kickoff"], sample["result_available_at"]))
            if (not parse_time(joint_model.metadata["training_as_of"]) < pred < kick <= label <= cutoff
                    or mid in joint_model.base_predictor.metadata["training_match_ids"]):
                raise ValueError("Past held-out outcome and reference-training exclusion required")
            gain = sample["delta"]
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain):
                raise ValueError("Finite historical gain required")
            identity = (mid, key)
            if identity in records:
                raise ValueError("Duplicate match/action/acquired-set supervision")
            records[identity] = {"match_id": mid, "kickoff": kick.isoformat(), "prediction_cutoff": pred.isoformat(),
                "result_available_at": label.isoformat(), "action_id": aid, "acquired_action_ids": acquired,
                "gain": float(gain), "generating_model_version": sample["oof_predictor_version"],
                "generating_model_sha256": joint_artifact_sha256}
            matches[mid].add(key)
        legal = {context_key(aid, acquired) for aid in ACTION_IDS for acquired in ([], [a for a in ACTION_IDS if a != aid])}
        if not matches or any(contexts != legal for contexts in matches.values()):
            raise ValueError("Every calibration match requires all four legal transitions")
        self.records = [records[k] for k in sorted(records)]
        grouped = defaultdict(lambda: defaultdict(list))
        for row in self.records:
            grouped[self._key(row["action_id"], row["acquired_action_ids"])][row["match_id"]].append(row)
        self.cells = {}
        for key, by_match in grouped.items():
            averaged = []
            for mid, items in by_match.items():
                if len({r["kickoff"] for r in items}) != 1:
                    raise ValueError("Conflicting kickoff for same match")
                averaged.append({"match_id": mid, "kickoff": items[0]["kickoff"],
                    "gain": math.fsum(r["gain"] for r in items)/len(items)})
            self.cells[key] = gain_summary(averaged, min_support=self.min_support)
        self.model = joint_model
        self.manifest = {"dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
            "training_as_of": cutoff.isoformat(), "diagnostics_as_of": cutoff.isoformat(),
            "training_match_ids": sorted(matches), "training_matches": len(matches), "training_samples": len(self.records),
            "information_mode": joint_model.metadata["information_mode"], "joint_model_version": joint_model.model_version,
            "joint_artifact_sha256": joint_artifact_sha256, "joint_training_as_of": joint_model.metadata["training_as_of"],
            "latest_training_label_available_at": max(r["result_available_at"] for r in self.records),
            "estimator_kind": VERSION, "erasure_mode": self.mode, "target": "empirical_fixed_predictor_query_gain",
            "weighting": "two recorded states averaged within match/action, then equal matches" if self.mode == "action_only" else "one gain per match/action/acquired-set, equal matches",
            "runtime_value_inputs": ["action_id"]+(["acquired_action_ids"] if self.mode == "acquired_set" else []),
            "model_evsi_usage": "none", "validation_usage": "diagnostic_only_no_parameter_selection", "metadata": metadata or {}}
        self.version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def estimate_numbers(self, action_id, acquired, *, risk_level="mean"):
        if risk_level != "mean":
            raise ValueError("Gain-prior controls support only mean decisions")
        key = self._key(action_id, acquired)
        summary = self.cells.get(key, gain_summary([], min_support=self.min_support))
        result = {k:v for k,v in summary.items() if k not in {"match_values", "week_influences"}}
        mean = summary["expected_gain"]
        return {**result, "action_id": action_id, "acquired_action_ids": sorted(acquired), "context_key": key,
            "model_evsi": None, "evsi_bin": None, "g_hat": mean, "decision_gain": mean,
            "lower_gain": None, "upper_gain": None, "uncertainty": summary["standard_error"], "risk_level": "mean",
            "gate_reason": "stop_no_support" if mean is None else "stop_nonpositive_update_gain" if mean <= 0 else "eligible",
            "uncertainty_kind": "descriptive_UTC_week_cluster_mean_se_diagnostic_only",
            "gain_semantics": "empirical_fixed_predictor_one_step_update_gain",
            "estimator_kind": VERSION, "erasure_mode": self.mode, "estimator_version": self.version}

    def _visible(self, state):
        if self.model is None or not self.manifest:
            raise ValueError("Fit and bind gain prior before inference")
        clean, acquired, attempted = visible_context(state, model=self.model, trained_as_of=self.manifest["trained_as_of"])
        if parse_time(self.manifest["latest_training_label_available_at"]) >= parse_time(clean["prediction_cutoff"]):
            raise ValueError("Training outcomes must strictly precede runtime prediction")
        return clean, acquired, attempted

    def estimate_value(self, state, candidates, *, prediction=None, risk_level="mean"):
        if risk_level != "mean":
            raise ValueError("Gain-prior controls support only mean decisions")
        _, acquired, attempted = self._visible(state)
        if prediction is not None:
            _probabilities(prediction["probabilities"], "Supplied prediction")  # Shape only; never a value input.
        output, seen = [], set()
        for action in candidates:
            aid = action["action_id"]
            if aid in seen or aid in acquired or aid in attempted:
                raise ValueError("Unique unpurchased action required")
            seen.add(aid)
            result = self.estimate_numbers(aid, acquired, risk_level=risk_level)
            result.update(cost_vector=_cost(action), joint_model_version=self.model.model_version)
            output.append(result)
        return output

    def plan(self, state, actions, budget, *, risk_level="mean", lambda_cost=None):
        if risk_level != "mean":
            raise ValueError("Gain-prior controls support only mean decisions")
        if not isinstance(actions, (list, tuple)):
            raise ValueError("Actions must be a sequence")
        _, acquired, attempted = self._visible(state)
        used = set(acquired) | attempted
        limits, spent = _budget(budget)
        weights = lambda_cost or {}
        if set(weights)-set(UNITS):
            raise ValueError("Unknown cost-conversion unit")
        lam = {u: _number(weights.get(u, 0.), "lambda "+u) for u in UNITS}
        for action in actions:
            if not isinstance(action, dict) or action.get("action_id") not in ACTION_IDS or type(action.get("available", True)) is not bool:
                raise ValueError("Modeled action identity and Boolean availability required")
            if action.get("evidence_id") is not None and (not isinstance(action["evidence_id"], str) or not action["evidence_id"]):
                raise ValueError("Evidence identity must be a nonempty inert string")
        used_evidence = {a["evidence_id"] for a in actions if a["action_id"] in used and a.get("evidence_id") is not None}
        output, seen = [], set()
        for action in sorted(actions, key=lambda a:a["action_id"]):
            aid = action["action_id"]
            if aid in seen:
                raise ValueError("Duplicate candidate action")
            seen.add(aid)
            cost = _cost(action)
            reason = ("action_unavailable" if not action.get("available", True) else "already_attempted" if aid in used else
                "evidence_already_observed" if action.get("evidence_id") in used_evidence else
                "physical_budget_exhausted" if any(spent[u]+cost[u] > cap+1e-12 for u,cap in limits.items()) else None)
            if reason:
                output.append({"action_id": aid, "feasible": False, "reason": reason})
                continue
            estimate = self.estimate_numbers(aid, acquired)
            penalty = math.fsum(lam[u]*cost[u] for u in UNITS)
            mean = estimate["expected_gain"]
            net = mean-penalty if mean is not None else None
            output.append({**estimate, "cost_vector": cost, "joint_model_version": self.model.model_version,
                "feasible": mean is not None, "reason": estimate["gate_reason"], "cost_penalty": penalty,
                "mean_net_gain": net, "lower_net_gain": None, "net_gain": net,
                "selection_score": round(net, 12) if net is not None else None})
        eligible = [r for r in output if r["feasible"] and r["selection_score"] > 0]
        selected = min(eligible, key=lambda r:(-r["selection_score"], r["action_id"])) if eligible else None
        return {"schema_version": VERSION, "estimator_kind": VERSION, "erasure_mode": self.mode,
            "estimator_version": self.version, "model_version": self.model.model_version,
            "decision": "acquire" if selected else "stop", "action_id": selected["action_id"] if selected else None,
            "reason": "positive_historical_mean_net_gain" if selected else "no_positive_supported_historical_mean_gain",
            "risk_level": "mean", "horizon": 1, "physical_budget": {"limits": limits, "spent": spent},
            "lambda_cost": lam, "selection_score_precision_decimal_places": 12, "candidates": output}

    def _payload(self):
        return {"schema": VERSION, "parameters": {"mode": self.mode, "min_support": self.min_support},
            "records": self.records, "cells": self.cells, "manifest": self.manifest}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self._payload(), "version": self.version}, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, model, model_sha256):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != VERSION or not isinstance(model, JointReturnPredictor):
            raise ValueError("Gain-prior and learned-joint schemas required")
        result = cls(**raw["parameters"])
        for name in ("records", "cells", "manifest"):
            setattr(result, name, raw[name])
        expected = VERSION+":"+sha256(_canonical(result._payload()).encode()).hexdigest()[:20]
        if raw.get("version") != expected:
            raise ValueError("Gain-prior artifact hash mismatch")
        if (result.manifest["joint_model_version"] != model.model_version or result.manifest["joint_artifact_sha256"] != model_sha256
                or result.manifest["joint_training_as_of"] != model.metadata["training_as_of"]
                or result.manifest["erasure_mode"] != result.mode):
            raise ValueError("Gain-prior reference binding mismatch")
        result.model, result.version = model, expected
        return result
