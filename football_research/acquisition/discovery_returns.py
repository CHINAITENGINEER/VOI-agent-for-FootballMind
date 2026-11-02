"""Learned menu transitions and a bounded two-acquisition text policy.

The first response law is the frozen TextReturnModel. Conditional on its
response token, a second, outcome-independent empirical law predicts which
request *classes* become visible. Historical request prototypes are used only
inside hypothetical planning; they are never operational candidate URLs.

This is the value of an explicitly restricted policy: after the first return,
choose the best supported, feasible one-step action or stop. Unknown children
remain unresolved, and missing transition support invalidates the numeric
two-step estimate. No semantic search reward is supplied.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .expected_voi import evaluate_candidate, unknown_candidate
from .joint_planner import UNITS, _budget, _cost, _number
from .state import parse_time
from .text_returns import TextReturnModel, ORIGINS, _action, _canonical, _digest, _p, _text


VERSION = "learned-discovery-two-step-v2"
LEGACY_VERSION = "learned-discovery-two-step-v1"
_MENU_FIELDS = {"action_id", "tool", "url", "query", "source", "available", "cost_status", "cost",
                "information_key", "group_ids", "version", "discovered_at", "available_at"}


class DiscoveryReturnModel:
    """Frozen empirical request-class transitions with finite-horizon VoI."""

    version = VERSION

    def __init__(self, text_model):
        if not isinstance(text_model, TextReturnModel) or not text_model.model_version:
            raise ValueError("A fitted frozen TextReturnModel is required")
        self.text_model = text_model
        self.artifact_schema = VERSION
        self.parameters, self.metadata, self.records, self.prototypes = {}, {}, [], {}
        self.model_version = ""
        self._contexts = None

    def _menu_action(self, raw):
        if not isinstance(raw, dict) or set(raw)-_MENU_FIELDS:
            raise ValueError("Discovery candidates contain only requests, identity and declared costs")
        if raw.get("group_ids", []) or type(raw.get("available", True)) is not bool:
            raise ValueError("Discovery candidates require Boolean availability and no structured groups")
        clean, channel, token, vector = self.text_model._encode_action(raw)
        cost = _cost(raw)
        if cost["requests"] < 1.:
            raise ValueError("Every discovery HTTP action must reserve at least one physical request")
        # The learned abstraction deliberately excludes fixture-specific URL
        # identifiers and any content not obtained at the discovery step.
        signature = {"channel": channel, "action_token": token, "cost": cost,
                     "cost_status": raw["cost_status"], "available": raw.get("available", True)}
        key = "request-class:"+_digest(signature)[:24]
        action = {**clean, "action_id": "hypothesis:"+key, "cost": cost,
                  "cost_status": raw["cost_status"], "available": raw.get("available", True)}
        return key, signature, action, vector

    def fit(self, records, *, as_of, dataset_version, evidence_origin="real_observational",
            min_matches=None, min_class_matches=None, max_menu_actions=32, max_menu_branches=64):
        self.artifact_schema = VERSION
        if evidence_origin not in ORIGINS or evidence_origin != self.text_model.metadata["evidence_origin"]:
            raise ValueError("Discovery and text model evidence origins must agree")
        as_of = parse_time(as_of)
        if as_of < parse_time(self.text_model.metadata["training_as_of"]):
            raise ValueError("Discovery training cutoff precedes its frozen text model")
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("Observed, independently settled discovery records required")
        min_matches = self.text_model.parameters["min_matches"] if min_matches is None else min_matches
        min_class_matches = (self.text_model.parameters["min_class_matches"]
                             if min_class_matches is None else min_class_matches)
        for value in (min_matches, min_class_matches):
            if _number(value, "Support threshold") <= 0:
                raise ValueError("Strictly positive support thresholds required")
        for value in (max_menu_actions, max_menu_branches):
            if type(value) is not int or not 1 <= value <= 128:
                raise ValueError("Finite menu limits must be integers between one and 128")
        self.parameters = {"min_matches": float(min_matches), "min_class_matches": float(min_class_matches),
                           "max_menu_actions": max_menu_actions, "max_menu_branches": max_menu_branches}
        text_bindings = {_digest({k: row[k] for k in ("match_id", "outcome", "prior_probabilities",
                         "history_key", "action_digest", "response_digest", "prediction_cutoff",
                         "result_available_at")}) for row in self.text_model.records}
        stored, seen, prototypes = [], set(), defaultdict(dict)
        for raw in records:
            if not isinstance(raw, dict) or raw.get("menu_complete") is not True:
                raise ValueError("An observed complete menu is required; absent menus cannot mean empty")
            menu = raw.get("discovered_actions")
            if not isinstance(menu, list) or len(menu) > max_menu_actions:
                raise ValueError("A finite observed discovered_actions list is required")
            match_id = _text(raw.get("match_id"), "Training match ID")
            cutoff, result_at = parse_time(raw["prediction_cutoff"]), parse_time(raw["result_available_at"])
            if not cutoff < result_at <= as_of:
                raise ValueError("Independent settlement must follow prediction and precede training")
            for field in ("decision_at", "observed_at", "available_at", "menu_observed_at"):
                if raw.get(field) is not None and parse_time(raw[field]) > cutoff:
                    raise ValueError("Future discovery/menu observations cannot train a pre-match policy")
            history = raw.get("history", [])
            if not isinstance(history, list) or len(history) > 31:
                raise ValueError("A finite actually acquired history is required")
            h = [self.text_model._history_entry(step["action"], step["response"]) for step in history]
            clean, channel, token, vector = self.text_model._encode_action(raw["action"])
            response_token = self.text_model._encode_response(raw["response"])
            if token is None or response_token is None or any(x["unsupported"] for x in h):
                raise ValueError("Discovery observations require frozen text-model vocabulary support")
            binding = {"match_id": match_id, "outcome": raw["outcome"],
                       "prior_probabilities": _p(raw["prior_probabilities"]),
                       "history_key": self.text_model._history_key(h), "action_digest": _digest(clean),
                       "response_digest": _digest(raw["response"]), "prediction_cutoff": cutoff.isoformat(),
                       "result_available_at": result_at.isoformat()}
            if _digest(binding) not in text_bindings:
                raise ValueError("Discovery record is not bound to an eligible frozen text training record")
            keys = []
            for candidate in menu:
                for field in ("available_at", "discovered_at"):
                    if candidate.get(field) is not None and parse_time(candidate[field]) > cutoff:
                        raise ValueError("A future candidate cannot enter the observed menu")
                key, signature, action, avector = self._menu_action(candidate)
                keys.append(key)
                prototypes[key][_digest(action)] = (signature, action, avector)
            if response_token in {"failure", "no_data"} and keys:
                raise ValueError("A failed/empty return cannot invent discovered candidates")
            row = {**binding, "channel": channel, "action_token": token,
                   "action_vector": vector.tolist(), "response_token": response_token,
                   "menu": sorted(set(keys))}
            identity = _digest(row)
            if identity not in seen:
                stored.append(row)
                seen.add(identity)
        chosen = {}
        for key, unique_entries in sorted(prototypes.items()):
            entries = list(unique_entries.values())
            # A medoid has an actual training request vector; inference does
            # not invent a future URL or fit a held-out request representation.
            supported = [x for x in entries if x[2] is not None]
            if supported:
                center = np.mean([x[2] for x in supported], axis=0)
                entry = min(supported, key=lambda x: (float(np.sum((x[2]-center)**2)), _canonical(x[1])))
            else:
                entry = min(entries, key=lambda x: _canonical(x[1]))
            chosen[key] = {"signature": entry[0], "action": entry[1], "supported_encoding": bool(supported)}
        self.records, self.prototypes, self._contexts = stored, chosen, None
        self.metadata = {"training_as_of": as_of.isoformat(), "trained_as_of": as_of.isoformat(),
                         "dataset_version": _text(dataset_version, "Dataset version"),
                         "text_model_version": self.text_model.model_version,
                         "training_match_ids": sorted({r["match_id"] for r in stored}),
                         "training_matches": len({r["match_id"] for r in stored}),
                         "training_records": len(stored), "evidence_origin": evidence_origin,
                         "production_ready": False, "calibration_status": "not_established",
                         "horizon": 2, "selection_scope": "supported_child_or_stop_policy_not_all_open_world_actions",
                         "menu_law": "Y_independent_given_frozen_response_token_p0_action_observed_history",
                         "menu_conditioning_limit": "child_kernel_ignores_menu_identity_conditional_on_parent_response_history",
                         "menu_retention_assumption": "current_unattempted_candidates_remain_available_additions_only_cap_removals_not_modeled",
                         "request_abstraction": "frozen_training_action_cluster_normalized_centroid_and_cost",
                         "request_value_mode": "training_cluster_centroid",
                         "runtime_rule": "hypothetical_training_requests_are_never_operational_candidates",
                         "unknown_policy": "missing_transition_unknown_supported_child_policy_only"}
        self.model_version = VERSION+":"+_digest(self._payload())[:24]
        return self

    def _transitions(self, state, action, response_token):
        _, channel, token, vector = self.text_model._encode_action(action)
        if token is None:
            return None, "unsupported_discovery_action", {}
        if self._contexts is None:
            self._contexts = defaultdict(list)
            for row in self.records:
                self._contexts[(row["channel"], _canonical(row["history_key"]), row["response_token"])].append(row)
        rows = self._contexts.get((channel, _canonical(self.text_model._history_key(state["history"])), response_token), [])
        weighted = []
        for row in rows:
            similarity = float(np.dot(vector, np.asarray(row["action_vector"])))
            if similarity+1e-12 < self.text_model.parameters["min_action_similarity"]:
                continue
            delta = np.asarray(row["prior_probabilities"])-np.asarray(state["prior_probabilities"])
            weight = similarity*math.exp(-float(np.dot(delta, delta))/(2*self.text_model.parameters["bandwidth"]**2))
            if weight > 0:
                weighted.append((row, weight))
        count = Counter(r["match_id"] for r, _ in weighted)
        weighted = [(r, w/count[r["match_id"]]) for r, w in weighted]
        by_match, classes, menus = defaultdict(float), [0., 0., 0.], defaultdict(float)
        for row, weight in weighted:
            by_match[row["match_id"]] += weight
            classes[row["outcome"]] += weight
            menus[tuple(row["menu"])] += weight
        mass = math.fsum(by_match.values())
        square = math.fsum(w*w for w in by_match.values())
        effective = mass*mass/square if square else 0.
        support = {"matching_matches": len(by_match), "effective_matches": effective,
                   "class_weight": classes, "max_match_weight": max(by_match.values(), default=0.),
                   "smoothing_counts_as_support": False, "menu_branches": len(menus)}
        if effective+1e-9 < self.parameters["min_matches"] or min(classes)+1e-9 < self.parameters["min_class_matches"]:
            return None, "insufficient_discovery_transition_support", support
        if len(menus) > self.parameters["max_menu_branches"]:
            return None, "discovery_transition_branch_limit", support
        return [{"menu": list(menu), "probability": weight/mass} for menu, weight in sorted(menus.items())], "supported", support

    def _hypothesis(self, state, action, branch):
        """Planning-only branch; never return it as an actually observed state."""
        clean, channel, token, _ = self.text_model._encode_action(action)
        history = {"action": clean, "channel": channel, "action_token": token,
                   "response_token": branch["branch_id"], "response_digest": "hypothetical:"+branch["branch_id"],
                   "unsupported": False}
        return {**deepcopy(state), "current_probabilities": list(branch["posterior"]),
                "history": deepcopy(state["history"])+[history]}

    def enumerate(self, state, action, *, budget, lambda_cost, current_actions=(), loss="log_loss"):
        if not self.model_version:
            raise ValueError("Fit or restore a discovery model first")
        state = self.text_model._state(state)
        if parse_time(state["prediction_cutoff"]) < parse_time(self.metadata["training_as_of"]):
            raise ValueError("Discovery model was trained after this prediction cutoff")
        if not isinstance(lambda_cost, dict) or set(lambda_cost)-set(UNITS):
            raise ValueError("Declared loss-conversion units required")
        weights = {u: _number(lambda_cost.get(u, 0), "Cost conversion") for u in UNITS}
        limits, spent = _budget(budget)
        physical = _cost(action)
        if physical["requests"] < 1.:
            raise ValueError("Every discovery HTTP action must reserve at least one physical request")
        price = math.fsum(weights[u]*physical[u] for u in UNITS)
        immediate = self.text_model.enumerate(state, action, cost=price, loss=loss)
        clean = immediate["action"]
        aid = action.get("action_id") or immediate["value"]["candidate_id"]
        attempted = {_digest(entry["action"]) for entry in state["history"]}
        current = []
        if not isinstance(current_actions, (list, tuple)):
            raise ValueError("Current real candidate menu must be a sequence")
        for raw in current_actions:
            c, _, _ = _action(raw)
            pcost = _cost(raw)
            if pcost["requests"] < 1.:
                raise ValueError("Every discovery HTTP action must reserve at least one physical request")
            if type(raw.get("available", True)) is not bool:
                raise ValueError("Known candidate availability must be Boolean")
            current.append(({**deepcopy(raw), "cost": pcost}, _digest(c)))
        after_spent = {u: spent[u]+physical[u] for u in UNITS}
        result = {"schema": self.artifact_schema, "model_version": self.model_version,
                  "text_model_version": self.text_model.model_version, "state_digest": _digest(state),
                  "action_digest": _digest(clean), "response_enumeration": immediate,
                  "immediate_evsi": immediate["value"]["gross_evsi"],
                  "immediate_cost": price, "expected_future_evsi": None, "expected_future_cost": None,
                  "expected_continuation_value": None, "expected_future_cost_vector": None,
                  "continuation_branches": [], "unresolved_child_ids": [], "unresolved_alternatives": False,
                  "value_scope": self.metadata["selection_scope"], "horizon": 2,
                  "production_ready": False, "evidence_origin": self.metadata["evidence_origin"]}

        def finish_unknown(reason):
            result.update(status="unknown", reason=reason,
                          value=unknown_candidate(aid, state["current_probabilities"], reason=reason,
                            cost=price, loss=loss, predictor_version=self.text_model.model_version).as_dict())
            result["decision_digest"] = _digest(result)
            return result

        if _digest(clean) in attempted or not action.get("available", True):
            return finish_unknown("first_action_unavailable_or_attempted")
        if any(after_spent[u] > cap+1e-12 for u, cap in limits.items()):
            return finish_unknown("physical_budget_exhausted")
        if immediate["status"] != "supported":
            return finish_unknown(immediate["reason"])
        if len(state["history"])+2 > 32:
            return finish_unknown("two_step_history_limit")
        # Every actual acquisition consumes an HTTP request. With no request
        # left, the continuation is provably infeasible, not statistically zero.
        no_next_request = "requests" in limits and after_spent["requests"]+1 > limits["requests"]+1e-12
        terminal, all_unknown = [], set()
        future_evsi, future_cost = 0., 0.
        future_vector = {u: 0. for u in UNITS}
        for branch in immediate["branches"]:
            if no_next_request:
                transitions, reason, support = [{"menu": [], "probability": 1.}], "no_remaining_request_budget", {}
            else:
                transitions, reason, support = self._transitions(state, action, branch["branch_id"])
            if transitions is None:
                result["continuation_branches"].append({"response_token": branch["branch_id"],
                    "status": "unknown", "reason": reason, "support": support})
                return finish_unknown(reason)
            hypothetical = self._hypothesis(state, action, branch)
            for transition in transitions:
                probability = branch["probability"]*transition["probability"]
                children = [] if no_next_request else [(c, identity, False) for c, identity in current]
                for key in transition["menu"]:
                    prototype = self.prototypes[key]
                    child = prototype["action"]
                    children.append((child, _digest(_action(child)[0]), True))
                feasible, unresolved, used = [], [], set()
                for child, identity, hypothetical_request in children:
                    if identity in used or identity in attempted or identity == _digest(clean):
                        continue
                    # A current unavailable request still overrides a matching
                    # historical prototype; observed availability wins.
                    used.add(identity)
                    if not child.get("available", True):
                        continue
                    cphysical = _cost(child)
                    if any(after_spent[u]+cphysical[u] > cap+1e-12 for u, cap in limits.items()):
                        continue
                    child_price = math.fsum(weights[u]*cphysical[u] for u in UNITS)
                    method = (self.text_model.enumerate_hypothetical_request_class
                              if hypothetical_request and self.metadata.get("request_value_mode") == "training_cluster_centroid"
                              else self.text_model.enumerate)
                    child_value = method(hypothetical, child, cost=child_price, loss=loss)
                    child_id = child.get("action_id") or "request:"+identity[:20]
                    if child_value["status"] != "supported":
                        unresolved.append(child_id)
                        continue
                    feasible.append({"child_id": child_id, "physical_cost": cphysical,
                                     "enumeration": child_value})
                best = min((c for c in feasible if c["enumeration"]["value"]["net_voi"] > 1e-12),
                           key=lambda c: (-c["enumeration"]["value"]["net_voi"], c["child_id"]), default=None)
                all_unknown.update(unresolved)
                branch_record = {"response_token": branch["branch_id"], "probability": probability,
                    "menu_probability": transition["probability"], "discovered_request_classes": transition["menu"],
                    "selected_child_id": best["child_id"] if best else None,
                    "decision": "query" if best else "stop_supported_subpolicy",
                    "unresolved_child_ids": sorted(set(unresolved)), "transition_support": support,
                    "gross_future_evsi": 0., "future_cost": 0., "continuation_value": 0.}
                if best:
                    chosen = best["enumeration"]
                    evsi, cost = chosen["value"]["gross_evsi"], chosen["value"]["cost"]
                    future_evsi += probability*evsi
                    future_cost += probability*cost
                    for u in UNITS:
                        future_vector[u] += probability*best["physical_cost"][u]
                    branch_record.update(gross_future_evsi=evsi, future_cost=cost, continuation_value=evsi-cost)
                    for child_branch in chosen["branches"]:
                        terminal.append({"branch_id": str(len(terminal)), "probability": probability*child_branch["probability"],
                                         "posterior": child_branch["posterior"]})
                else:
                    terminal.append({"branch_id": str(len(terminal)), "probability": probability, "posterior": branch["posterior"]})
                result["continuation_branches"].append(branch_record)
        value = evaluate_candidate(aid, state["current_probabilities"], terminal, cost=price+future_cost,
                                   loss=loss, predictor_version=self.text_model.model_version,
                                   response_model_version=self.model_version, calibration_cutoff=self.metadata["training_as_of"])
        result.update(expected_future_evsi=future_evsi, expected_future_cost=future_cost,
                      expected_future_cost_vector=future_vector, expected_continuation_value=future_evsi-future_cost,
                      unresolved_child_ids=sorted(all_unknown), unresolved_alternatives=bool(all_unknown))
        if all_unknown and (value.net_voi is None or value.net_voi <= 1e-12):
            result["supported_subpolicy_value"] = value.as_dict()
            return finish_unknown("nonpositive_supported_subpolicy_with_unresolved_children")
        result.update(status="supported", reason="computed_two_step_supported_subpolicy", value=value.as_dict())
        result["decision_digest"] = _digest(result)
        return result

    def _payload(self):
        return {"schema": self.artifact_schema, "text_model_version": self.text_model.model_version,
                "parameters": self.parameters, "metadata": self.metadata,
                "records": self.records, "prototypes": self.prototypes}

    def save(self, path):
        if not self.model_version:
            raise ValueError("Cannot save an unfitted discovery model")
        payload = self._payload()
        Path(path).write_text(json.dumps({**payload, "model_version": self.model_version,
            "content_sha256": _digest(payload)}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, text_model):
        def reject(value):
            raise ValueError("Non-finite JSON constant: "+value)
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON field: "+key)
                result[key] = value
            return result
        data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=pairs)
        keys = {"schema", "text_model_version", "parameters", "metadata", "records", "prototypes", "model_version", "content_sha256"}
        if not isinstance(data, dict) or set(data) != keys or data["schema"] not in {LEGACY_VERSION, VERSION}:
            raise ValueError("Unexpected frozen discovery model schema")
        payload = {k: v for k, v in data.items() if k not in {"model_version", "content_sha256"}}
        if data["content_sha256"] != _digest(payload) or data["model_version"] != data["schema"]+":"+_digest(payload)[:24]:
            raise ValueError("Frozen discovery model content mismatch")
        if (data["text_model_version"] != text_model.model_version or
                data["metadata"].get("text_model_version") != text_model.model_version):
            raise ValueError("Discovery model is bound to a different frozen text model")
        if data["metadata"].get("evidence_origin") != text_model.metadata["evidence_origin"]:
            raise ValueError("Discovery and text model evidence origins must agree")
        if data["metadata"].get("evidence_origin") not in ORIGINS or data["metadata"].get("production_ready") is not False:
            raise ValueError("Unvalidated discovery model cannot claim production readiness")
        model = cls(text_model)
        model.artifact_schema = data["schema"]
        model.parameters, model.metadata = data["parameters"], data["metadata"]
        model.records, model.prototypes, model.model_version = data["records"], data["prototypes"], data["model_version"]
        return model


__all__ = ["DiscoveryReturnModel", "VERSION"]
