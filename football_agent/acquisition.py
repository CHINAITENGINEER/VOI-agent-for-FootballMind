"""Public-state-only active acquisition execution and durable value traces.

Prediction and gain estimation belong to the research backend; this module
only validates, budgets, selects, executes and journals their actual outputs.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
from math import isfinite
from queue import Queue, Empty
from threading import Thread
import time
from uuid import uuid4

from .audit import RunStore, digest, redact
from .types import now, utc


STRATEGIES = {"fixed", "pure_llm", "all_candidates", "llm_gate"}
STRATEGY_ALIASES = {
    "fixed-order": "fixed",
    "numeric-all-candidates": "all_candidates",
    "llm-proposal+numeric-gate": "llm_gate",
    "pure-llm": "pure_llm",
}
RISK_MODES = {"mean", "conservative"}
PUBLIC_FIELDS = {"episode_id", "match_id", "prediction_cutoff", "information_mode", "base_facts", "groups", "mask",
                 "evidence_versions", "quality", "action_history", "budget", "state_version"}


def public_payload(value):
    forbidden = {"target_label", "target_result", "label", "outcome", "hidden_groups", "hidden_values",
                 "future_result", "full_snapshot", "features_json", "archive_root", "dataset_path",
                 "parquet_path", "source_path", "model_artifact"}
    def visit(item):
        if isinstance(item, dict):
            if set(item) & forbidden:
                raise ValueError("Hidden/label/file access material in public tool output")
            for child in item.values(): visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(value)
    digest(value)
    return value


def public_state(value):
    """Fail closed if a boundary returns hidden fields or unacquired groups."""
    if not isinstance(value, dict) or set(value) - PUBLIC_FIELDS:
        raise ValueError("VisibleState contains fields outside the public acquisition boundary")
    result = deepcopy(value)
    mask, groups = result.get("mask", {}), result.get("groups", {})
    if not isinstance(mask, dict) or not isinstance(groups, dict) or any(type(v) is not bool for v in mask.values()):
        raise ValueError("VisibleState requires a boolean group mask")
    if any(not mask.get(group, False) for group in groups):
        raise ValueError("Unacquired group crossed the visible-state boundary")
    if not result.get("match_id") or not result.get("prediction_cutoff"):
        raise ValueError("VisibleState lacks match/cutoff identity")
    utc(result["prediction_cutoff"])
    forbidden = {"target_label", "target_result", "label", "outcome", "hidden_groups", "full_snapshot", "features_json",
                 "archive_root", "dataset_path", "parquet_path", "source_path", "model_artifact"}
    def visit(item):
        if isinstance(item, dict):
            if set(item) & forbidden:
                raise ValueError("Hidden/label/file access material in VisibleState")
            for child in item.values(): visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(result)
    digest(result)  # JSON/finite validation before any model/proposer call.
    return result


@dataclass(frozen=True)
class AcquisitionBudget:
    max_actions: int = 6
    max_simulated_cost: float = 3.0
    max_online_reserve: float = 1.0
    max_llm_calls: int = 5
    max_compute_calls: int = 30
    max_elapsed_seconds: float = 120.0
    max_tool_seconds: float = 20.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError("Invalid budget: " + key)
        for key in ("max_actions", "max_llm_calls", "max_compute_calls"):
            if type(getattr(self, key)) is not int:
                raise ValueError("Integer budget required: " + key)
        if self.max_llm_calls > 5 or self.max_tool_seconds <= 0:
            raise ValueError("LLM cap is five; tool timeout must be positive")


class AcquisitionWorkflow:
    version = "visible-acquisition-executor-v1"

    def __init__(self, backend, store_dir="var/parallel/agent/round3/runs", proposer=None):
        self.backend, self.proposer = backend, proposer
        self.store = RunStore(store_dir)

    def configuration(self):
        return {"executor_version": self.version, "backend": self.backend.configuration(),
                "proposer": self.proposer.configuration() if self.proposer else {"status": "NEEDS_CONFIGURATION"}}

    def start(self, match_id, cutoff, strategy="all_candidates", budget=None, seed=0, risk_mode="mean"):
        strategy = STRATEGY_ALIASES.get(strategy, strategy)
        if strategy not in STRATEGIES or type(seed) is not int or risk_mode not in RISK_MODES:
            raise ValueError("Unknown strategy or invalid seed")
        limits = budget if isinstance(budget, AcquisitionBudget) else AcquisitionBudget(**(budget or {}))
        proposer_config = self.proposer.configuration() if self.proposer else {"status": "NEEDS_CONFIGURATION"}
        if strategy == "llm_gate":
            evidence_class = "llm_proposer_numeric_gate"
        elif strategy == "pure_llm":
            evidence_class = "pure_llm" if proposer_config.get("configured") else "pure_llm_unconfigured"
        elif strategy == "all_candidates":
            evidence_class = "numeric_gate"
        else:
            evidence_class = "fixed_order"
        run = {"schema_version": "agent-acquisition-v1", "run_id": str(uuid4()), "match_id": match_id,
               "prediction_cutoff": utc(cutoff).isoformat(), "strategy": strategy, "risk_mode": risk_mode, "seed": seed,
               "state": "INITIALIZED", "reason": "INITIALIZED", "visible_state": None, "prediction": None,
               "candidates": [], "history": [], "events": [], "pending_call": None, "attempted": {}, "cooldowns": {},
               "evidence": {"class": evidence_class, "proposer": "llm" if strategy in {"llm_gate", "pure_llm"} else "none",
                             "numeric_gate": strategy in {"all_candidates", "llm_gate"},
                             "pure_llm_configured": bool(proposer_config.get("configured")) if strategy == "pure_llm" else None,
                             "environment_mode": self.backend.configuration().get("mode", "unknown"),
                             "provider": proposer_config.get("provider"),
                             "provider_configured": bool(proposer_config.get("configured")),
                             "scope": "execution_audit_only_not_effectiveness_evidence"},
               "proposal_audit": {"calls": 0, "valid_calls": 0, "invalid_calls": 0,
                                  "duplicate_or_invalid_calls": 0, "proposal_items": 0,
                                  "legal_parameter_items": 0, "legal_parameter_rate": None,
                                  "duplicate_or_invalid_call_rate": None},
               "stop_audit": {"status": "running", "risk_mode": risk_mode,
                              "decision_metric": "conservative_net_gain" if risk_mode == "conservative" else "net_gain",
                              "threshold": 0.0},
               "configuration": self.configuration(), "budget": {**asdict(limits), "actions": 0,
               "simulated_cost": 0.0, "online_reserve": 0.0, "llm_calls": 0, "compute_calls": 0,
               "elapsed_seconds": 0.0, "usage": [], "cost_ledger": []}, "created_at": now()}
        with self.store.lock(run["run_id"]):
            self._event(run, "initialized", {"strategy": strategy, "risk_mode": risk_mode, "seed": seed})
            result = self._call(run, "initialize_environment", lambda: self.backend.start(match_id, cutoff, seed), {})
            if result is not None:
                try:
                    run["visible_state"] = self._checked_state(run, result)
                    run["state"] = "READY"
                    self._event(run, "visible_state_initialized", {"mask": run["visible_state"]["mask"]})
                except (ValueError, TypeError) as exc:
                    self._finish(run, "DEFERRED", "INVALID_VISIBLE_STATE", str(exc))
            return deepcopy(run)

    def _checked_state(self, run, value):
        result = public_state(value)
        if result["match_id"] != run["match_id"] or utc(result["prediction_cutoff"]) != utc(run["prediction_cutoff"]):
            raise ValueError("Public state changed match/cutoff; start an explicitly linked episode")
        if run.get("visible_state") and result.get("information_mode") != run["visible_state"].get("information_mode"):
            raise ValueError("Public state changed information mode")
        return result

    def get_run(self, run_id):
        with self.store.lock(run_id):
            run = self.store.load(run_id)
            if run.get("pending_call"):
                run.setdefault("unreconciled_calls", []).append(run["pending_call"])
                run["pending_call"] = None
                self._finish(run, "DEFERRED", "INTERRUPTED_CALL_REQUIRES_RECONCILIATION")
            return deepcopy(run)

    def run_to_completion(self, run_id):
        with self.store.lock(run_id):
            while True:
                run = self.get_run(run_id)
                if run["state"] in {"STOPPED", "DEFERRED", "WAITING"}: return run
                self.step(run_id)

    def step(self, run_id):
        with self.store.lock(run_id):
            run = self.get_run(run_id)
            if run["state"] == "WAITING" and run.get("reason") == "ACTION_BACKOFF":
                if run.get("cooldowns") and all(utc(deadline) > utc(now()) for deadline in run["cooldowns"].values()):
                    return run
                run["state"] = "PREDICTED"
                self._event(run, "action_backoff_elapsed", {})
            if run["state"] in {"STOPPED", "DEFERRED", "WAITING"}: return run
            if digest(self.configuration()) != digest(run["configuration"]):
                self._finish(run, "DEFERRED", "RUN_CONFIGURATION_CHANGED")
                return run
            visible = self._checked_state(run, run["visible_state"])
            # Budget is public; consumers receive no journal, backend or config.
            visible["budget"] = {k: v for k, v in run["budget"].items() if k not in {"cost_ledger", "usage"}}
            if run["prediction"] is None or run["state"] == "ACQUIRED":
                prediction = self._call(run, "partial_predict", lambda: self.backend.predict(deepcopy(visible)), {"visible_state": visible})
                if prediction is None: return run
                if not self._valid_prediction(prediction):
                    self._finish(run, "DEFERRED", "INVALID_PARTIAL_PREDICTION")
                    return run
                previous = run.get("prediction")
                run["prediction"], run["state"] = prediction, "PREDICTED"
                self._event(run, "prediction_updated", {"before": previous, "after": prediction, "mask": visible["mask"]})
                return run
            directory = self._call(run, "action_catalog", lambda: self.backend.actions(deepcopy(visible)), {"visible_state": visible})
            if directory is None: return run
            try:
                candidates = self._candidates(run, directory)
            except (ValueError, KeyError, TypeError) as exc:
                self._finish(run, "DEFERRED", "INVALID_ACTION_CATALOG", str(exc)); return run
            run["candidates"] = candidates
            legal = [a for a in candidates if a["available"]]
            self._event(run, "candidates_listed", {"candidates": candidates})
            if not legal:
                if any(a.get("unavailable_reason") == "RETRY_BACKOFF" for a in candidates):
                    self._finish(run, "WAITING", "ACTION_BACKOFF"); return run
                reason = "BUDGET_EXHAUSTED" if any(a.get("unavailable_reason") == "BUDGET_EXHAUSTED" for a in candidates) else "NO_LEGAL_ACTIONS"
                self._finish(run, "STOPPED", reason); return run
            proposed = legal
            if run["strategy"] in {"pure_llm", "llm_gate"}:
                if (self.proposer is None or self.proposer.configuration().get("status") == "NEEDS_CONFIGURATION"
                        or self.proposer.configuration().get("configured") is False):
                    self._finish(run, "DEFERRED", "LLM_NEEDS_CONFIGURATION"); return run
                planner_state = {**deepcopy(visible), "current_prediction": deepcopy(run["prediction"])}
                proposal = self._call(run, "llm_propose", lambda: self.proposer.propose(planner_state, deepcopy(legal),
                    request_id=run["pending_call"]["request_id"], remaining_calls=run["budget"]["max_llm_calls"] - run["budget"]["llm_calls"] + 1),
                                      {"visible_state": planner_state, "actions": legal})
                if proposal is None: return run
                if proposal.get("status") != "OK":
                    self._record_proposal_audit(run, {"status": proposal.get("status", "FAILED"),
                        "raw_count": len(proposal.get("proposals", [])) if isinstance(proposal.get("proposals"), list) else 0,
                        "legal_count": 0, "duplicate_or_invalid": 1,
                        "legal_parameter_rate": 0.0, "invalid_reason": proposal.get("reason", "")})
                    self._event(run, "llm_proposal_audited", {"status": proposal.get("status", "FAILED"),
                        "valid": False, "legal_parameter_rate": 0.0, "duplicate_or_invalid": True,
                        "evidence_class": run.get("evidence", {}).get("class", "legacy_unclassified")})
                    self._finish(run, "DEFERRED", "LLM_" + str(proposal.get("status", "FAILED"))); return run
                raw_items = proposal.get("proposals", []) if isinstance(proposal.get("proposals"), list) else []
                proposal_audit = {"status": "PENDING_VALIDATION", "raw_count": len(raw_items), "legal_count": 0,
                                  "legal_parameter_rate": None,
                                  "duplicate_or_invalid": 0, "evidence_class": run.get("evidence", {}).get("class", "legacy_unclassified")}
                proposal["proposal_audit"] = proposal_audit
                self._event(run, "llm_proposed", proposal)
                try:
                    ids = []
                    if not isinstance(proposal.get("proposals", []), list):
                        raise ValueError("Proposals must be a list")
                    for item in proposal.get("proposals", []):
                        if not isinstance(item, dict):
                            raise ValueError("Proposal item must be an object")
                        action = next(a for a in legal if a["action_id"] == item["action_id"])
                        if item.get("parameters", {}) != action.get("parameters", {}):
                            raise ValueError("Proposed parameters differ from catalog executable instance")
                        ids.append(item["action_id"])
                    proposed = [next(a for a in legal if a["action_id"] == mid) for mid in dict.fromkeys(ids)]
                except (KeyError, TypeError, StopIteration, ValueError):
                    self._record_proposal_audit(run, {"status": "ILLEGAL_PROPOSAL", "raw_count": len(raw_items),
                        "legal_count": 0, "duplicate_or_invalid": 1, "legal_parameter_rate": 0.0})
                    self._event(run, "llm_proposal_audited", {"status": "ILLEGAL_PROPOSAL", "valid": False,
                        "legal_parameter_rate": 0.0, "duplicate_or_invalid": True,
                        "evidence_class": run.get("evidence", {}).get("class", "legacy_unclassified")})
                    self._finish(run, "DEFERRED", "ILLEGAL_LLM_PROPOSAL"); return run
                proposal_audit.update(status="OK", legal_count=len(raw_items), legal_parameter_rate=1.0 if raw_items else None)
                proposal_audit["duplicate_items"] = len(ids) - len(set(ids))
                proposal_audit["duplicate_or_invalid"] = int(len(ids) != len(set(ids)))
                self._record_proposal_audit(run, proposal_audit)
                self._event(run, "llm_proposal_audited", proposal_audit)
                if proposal.get("stop") and run["strategy"] == "pure_llm":
                    self._finish(run, "STOPPED", "LLM_STOP"); return run
                if proposal.get("stop"): proposed = []
                if not proposed and run["strategy"] == "pure_llm":
                    self._finish(run, "STOPPED", "LLM_NO_ACTION"); return run
            if run["strategy"] in {"all_candidates", "llm_gate"}:
                # Small catalogs are enumerated to quantify proposal omissions.
                values = self._call(run, "estimate_value", lambda: self.backend.estimate(deepcopy(visible), deepcopy(legal)),
                                    {"visible_state": visible, "actions": legal})
                if values is None: return run
                valid = self._merge_values(run, legal, values)
                offered = {a["action_id"] for a in proposed}
                gain_key = "conservative_net_gain" if run.get("risk_mode") == "conservative" else "net_gain"
                omitted = [v["action_id"] for v in valid if v.get(gain_key) is not None and v[gain_key] > 0 and v["action_id"] not in offered]
                supported = [v for v in valid if v.get(gain_key) is not None]
                positive = [v for v in supported if v[gain_key] > 0]
                positive_ids = {v["action_id"] for v in positive}
                proposal_order = [a["action_id"] for a in proposed]
                recall_at_k = {}
                for k in (1, 3, 5):
                    prefix = set(proposal_order[:k])
                    recall_at_k[str(k)] = (len(prefix & positive_ids) / len(positive_ids)) if positive_ids else None
                ranked_positive = sorted(positive, key=lambda v: (-v[gain_key], v["action_id"]))
                top_action = ranked_positive[0]["action_id"] if ranked_positive else None
                top_rank = proposal_order.index(top_action) + 1 if top_action in proposal_order else None
                proposal_audit = {"legal_count": len(legal), "proposed_count": len(offered),
                                  "supported_count": len(supported), "positive_count": len(positive),
                                  "positive_omitted": omitted,
                                  "positive_recall": (len(positive) - len(omitted)) / len(positive) if positive else None,
                                  "proposal_recall_at_k": recall_at_k,
                                  "recall_reference": "positive_numeric_estimates_not_realized_oracle",
                                  "recall_applicable": run["strategy"] == "llm_gate",
                                  "top_value_action": top_action, "top_value_action_rank": top_rank,
                                  "risk_mode": run.get("risk_mode", "mean"),
                                  "decision_metric": gain_key,
                                  "evidence_class": run.get("evidence", {}).get("class", "legacy_unclassified")}
                self._event(run, "value_estimated", {"candidates": run["candidates"],
                    "positive_actions_omitted_by_proposer": omitted, "proposal_audit": proposal_audit})
                if not valid:
                    self._finish(run, "STOPPED", "ESTIMATE_UNAVAILABLE"); return run
                if not proposed:
                    self._finish(run, "STOPPED", "LLM_PROPOSAL_STOP_AFTER_VALUE_AUDIT"); return run
                eligible = [v for v in valid if v["action_id"] in offered]
                if not eligible:
                    self._finish(run, "STOPPED", "ESTIMATE_UNAVAILABLE_FOR_PROPOSALS"); return run
                selected_value = min(eligible, key=lambda v: (-(v.get(gain_key) if v.get(gain_key) is not None else float("-inf")), v["action_id"]))
                selected_gain = selected_value.get(gain_key)
                if selected_gain is None:
                    self._finish(run, "STOPPED", "ESTIMATE_UNAVAILABLE"); return run
                if selected_gain <= 0:
                    self._finish(run, "STOPPED", "VALUE_STOP_NONPOSITIVE_CONSERVATIVE_GAIN" if gain_key.startswith("conservative") else "VALUE_STOP_NONPOSITIVE_NET_GAIN"); return run
                chosen = next(a for a in legal if a["action_id"] == selected_value["action_id"])
            else:
                chosen = proposed[0]
            self._event(run, "action_selected", {"action_id": chosen["action_id"], "parameters": chosen.get("parameters", {}),
                                                 "strategy": run["strategy"]})
            result = self._call(run, "acquire", lambda: self.backend.acquire(deepcopy(visible), chosen["action_id"],
                deepcopy(chosen.get("parameters", {})), run["pending_call"]["request_id"]),
                {"visible_state": visible, "action": chosen}, action=chosen)
            if result is None: return run
            # A failure invalidates this action instance, never the whole plan.
            run["attempted"][chosen["execution_key"]] = {"status": result.get("status", "FAILED"), "at": now()}
            delay = result.get("retry_after_seconds")
            if type(delay) in {int, float} and isfinite(delay) and delay > 0:
                run.setdefault("cooldowns", {})[digest([chosen["action_id"], chosen.get("parameters", {})])] = (
                    utc(now()) + timedelta(seconds=delay)).isoformat()
            history = {"action_id": chosen["action_id"], "parameters": chosen.get("parameters", {}),
                       "execution_key": chosen["execution_key"], "result": result}
            try:
                updated = self._checked_state(run, result.get("visible_state", visible))
                changed = digest({"groups": updated["groups"], "mask": updated["mask"]}) != digest({"groups": visible["groups"], "mask": visible["mask"]})
                if result.get("status") not in {"OK", "ACQUIRED", "SUCCESS"} and changed:
                    raise ValueError("Failed action revealed new groups")
                if result.get("status") in {"OK", "ACQUIRED", "SUCCESS"}:
                    newly = {g for g in updated["groups"] if updated["groups"].get(g) != visible["groups"].get(g)}
                    if newly - set(chosen.get("group_ids", [])):
                        raise ValueError("Acquisition revealed groups outside the selected action")
                run["visible_state"] = updated
                run["state"] = "ACQUIRED" if changed else "PREDICTED"
                run["reason"] = "EVIDENCE_ACQUIRED" if changed else "ACTION_FINISHED_RESELECT"
                history["evidence_changed"] = changed
                run["history"].append(history)
                self._event(run, "acquisition_result", history)
                if any(run["budget"][account] > run["budget"]["max_" + account] + 1e-12
                       for account in ("simulated_cost", "online_reserve")):
                    self._finish(run, "STOPPED", "BUDGET_ACTUAL_COST_OVERRUN")
            except (KeyError, TypeError, ValueError) as exc:
                self._finish(run, "DEFERRED", "INVALID_ACQUISITION_RESULT", str(exc))
            return deepcopy(run)

    def _candidates(self, run, directory):
        if not isinstance(directory, list): raise ValueError("Catalog must be a list")
        seen, candidates = set(), []
        for raw in directory:
            item = deepcopy(raw)
            if set(item) & {"values", "hidden_values", "target_label", "future_result", "success_status", "full_snapshot"}:
                raise ValueError("Action catalog contains unknown return contents")
            ident = item["action_id"]
            if not isinstance(ident, str) or not ident or ident in seen: raise ValueError("Action IDs must be unique")
            seen.add(ident)
            item.setdefault("parameters", {})
            item["execution_key"] = digest([ident, item["parameters"], item.get("version"), item.get("evidence_version"),
                                           item.get("refresh_window"), run["prediction_cutoff"]])
            reason = None
            if item.get("status", "AVAILABLE") not in {"AVAILABLE", "OK", "available"} or item.get("available") is False:
                reason = item.get("reason", "UNAVAILABLE")
            elif item["execution_key"] in run["attempted"]: reason = "ALREADY_ATTEMPTED_THIS_VERSION"
            elif utc(run.get("cooldowns", {}).get(digest([ident, item["parameters"]]), "1970-01-01T00:00:00Z")) > utc(now()):
                reason = "RETRY_BACKOFF"
            elif not self._affordable(run, item): reason = "BUDGET_EXHAUSTED"
            item["available"] = reason is None
            item["unavailable_reason"] = reason
            candidates.append(item)
        return candidates

    @staticmethod
    def _action_cost(action):
        cost = action.get("cost") or {}
        amount = cost.get("amount")
        status = cost.get("status", "unknown")
        if amount is None: amount = cost.get("reserve", 1.0)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not isfinite(amount) or amount < 0:
            raise ValueError("Action cost must be finite and nonnegative")
        return float(amount), "simulated_cost" if status == "simulated" else "online_reserve"

    def _affordable(self, run, action):
        amount, account = self._action_cost(action)
        b = run["budget"]
        return b["actions"] < b["max_actions"] and b[account] + amount <= b["max_" + account] + 1e-12

    def _merge_values(self, run, legal, values):
        if isinstance(values, dict): values = values.get("values", values.get("candidates", []))
        if not isinstance(values, list): return []
        valid, seen = [], set()
        for value in values:
            if not isinstance(value, dict):
                continue
            item = deepcopy(value)
            ident = item.get("action_id")
            if ident in seen or ident not in {a["action_id"] for a in legal}: return []
            seen.add(ident)
            for candidate in run["candidates"]:
                if candidate["action_id"] == ident: candidate["value"] = item
            if item.get("status", "OK") != "OK": continue
            if item.get("support_status") in {"no_support", "parent_only"}: continue
            if item.get("gate_reason") in {"stop_no_support", "stop_insufficient_support"}: continue
            gain = item.get("expected_loss_improvement", item.get("expected_gain"))
            cost = item.get("cost_in_loss_units")
            net = item.get("net_gain")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in (gain, cost, net)): continue
            if cost < 0 or abs(net - (gain - cost)) > 1e-8: continue
            if not item.get("estimator_version", item.get("version")): continue
            item["expected_loss_improvement"] = gain
            lower = item.get("lower_gain")
            if isinstance(lower, (int, float)) and not isinstance(lower, bool) and isfinite(lower):
                item["conservative_net_gain"] = lower - cost
            else:
                item["conservative_net_gain"] = None
            for candidate in run["candidates"]:
                if candidate["action_id"] == ident: candidate["value"] = item
            valid.append(item)
        return valid

    @staticmethod
    def _record_proposal_audit(run, record):
        """Accumulate proposal validation evidence without exposing provider text."""
        audit = run.setdefault("proposal_audit", {"calls": 0, "valid_calls": 0, "invalid_calls": 0,
                                                   "duplicate_or_invalid_calls": 0, "proposal_items": 0,
                                                   "legal_parameter_items": 0})
        audit["calls"] += 1
        raw = record.get("raw_count", 0) if isinstance(record.get("raw_count", 0), int) else 0
        legal = record.get("legal_count", 0) if isinstance(record.get("legal_count", 0), int) else 0
        invalid = record.get("duplicate_or_invalid", 0)
        invalid = int(invalid) if isinstance(invalid, (int, bool)) else 0
        audit["proposal_items"] += max(0, raw)
        audit["legal_parameter_items"] += max(0, min(raw, legal))
        audit["duplicate_or_invalid_calls"] += max(0, invalid)
        if record.get("status") == "OK" and invalid == 0:
            audit["valid_calls"] += 1
        else:
            audit["invalid_calls"] += 1
        if audit["proposal_items"]:
            audit["legal_parameter_rate"] = audit["legal_parameter_items"] / audit["proposal_items"]
        audit["duplicate_or_invalid_call_rate"] = audit["duplicate_or_invalid_calls"] / audit["calls"]

    def _stop_audit(self, run, state, reason, detail=None):
        metric = "conservative_net_gain" if run.get("risk_mode") == "conservative" else "net_gain"
        audit = {"status": "stopped" if state == "STOPPED" else "deferred", "state": state,
                 "reason": reason, "risk_mode": run.get("risk_mode", "mean"),
                 "decision_metric": metric, "threshold": 0.0,
                 "basis": "risk_gate" if reason.startswith("VALUE_STOP_") else "workflow"}
        audit["stop_regret"] = None
        audit["stop_regret_status"] = "offline_counterfactual_reference_required"
        values = [event.get("data", {}) for event in run.get("events", []) if event.get("type") == "value_estimated"]
        if values:
            latest = values[-1]
            candidates = latest.get("candidates", [])
            scored = [item.get("value", {}) for item in candidates if isinstance(item, dict) and isinstance(item.get("value"), dict)]
            gains = [item.get(metric) for item in scored if isinstance(item.get(metric), (int, float))]
            audit.update({"evaluated_candidates": len(scored), "positive_candidates": sum(g > 0 for g in gains),
                          "max_observed_gain": max(gains) if gains else None,
                          "last_proposal_audit": deepcopy(latest.get("proposal_audit"))})
        return audit

    @staticmethod
    def _valid_prediction(prediction):
        if not isinstance(prediction, dict): return False
        p = prediction.get("probabilities", prediction)
        if isinstance(p, list): values = p
        elif isinstance(p, dict):
            keys = ("home", "draw", "away") if "home" in p else ("home_win_probability", "draw_probability", "away_win_probability")
            values = [p.get(k) for k in keys]
        else: return False
        return len(values) == 3 and all(type(v) in {int, float} and isfinite(v) and 0 <= v <= 1 for v in values) and abs(sum(values) - 1) < 1e-8

    def _call(self, run, name, function, inputs, action=None):
        b = run["budget"]
        counter = "llm_calls" if name == "llm_propose" else "actions" if name == "acquire" else "compute_calls"
        if b[counter] >= b["max_" + counter] or b["elapsed_seconds"] >= b["max_elapsed_seconds"]:
            self._finish(run, "STOPPED", "BUDGET_EXHAUSTED_" + counter.upper()); return None
        if name == "llm_propose":
            reserve = float((run["configuration"].get("proposer") or {}).get("reserve_per_call", 1.0))
            if b["online_reserve"] + reserve > b["max_online_reserve"]:
                self._finish(run, "STOPPED", "BUDGET_EXHAUSTED_LLM_RESERVE"); return None
            b["online_reserve"] += reserve
            b["cost_ledger"].append({"kind": "llm_unknown_fee_reservation", "amount": reserve,
                                      "unit": "local_reservation_unit_not_currency", "actual": None})
        if action:
            amount, account = self._action_cost(action)
            b[account] += amount
            b["cost_ledger"].append({"kind": "acquisition_reservation", "action_id": action["action_id"],
                "amount": amount, "account": account, "declared_cost": action.get("cost"), "actual": None})
        b[counter] += 1
        request_id = str(uuid4())
        input_ref = self.store.put_blob(run["run_id"], inputs)
        run["pending_call"] = {"request_id": request_id, "tool": name, "input_ref": input_ref,
                               "reservation_charged": True}
        self._event(run, "tool_started", run["pending_call"])
        queue, started = Queue(maxsize=1), time.perf_counter()
        def invoke():
            try: queue.put((function(), None))
            except Exception as exc: queue.put((None, {"category": type(exc).__name__, "message": str(exc)}))
        Thread(target=invoke, daemon=True).start()
        try:
            result, error = queue.get(timeout=min(b["max_tool_seconds"], b["max_elapsed_seconds"] - b["elapsed_seconds"]))
        except Empty:
            b["elapsed_seconds"] += time.perf_counter() - started
            self._finish(run, "DEFERRED", "CALL_TIMEOUT_REQUIRES_RECONCILIATION")
            return None
        b["elapsed_seconds"] += time.perf_counter() - started
        if error:
            if name == "acquire": result = {"status": "FAILED", "error": error, "visible_state": inputs["visible_state"]}
            else:
                run["pending_call"] = None
                self._finish(run, "DEFERRED", "TOOL_FAILED_" + name.upper(), error); return None
        try:
            # Validate public returns before storing anything from an environment.
            public_payload(result)
            if name == "initialize_environment": result = public_state(result)
            if name == "acquire" and "visible_state" in result: result["visible_state"] = public_state(result["visible_state"])
            output_ref = self.store.put_blob(run["run_id"], result)
        except (ValueError, TypeError) as exc:
            run["pending_call"] = None
            self._finish(run, "DEFERRED", "INVALID_TOOL_PUBLIC_OUTPUT", str(exc)); return None
        if name == "llm_propose": b["usage"].append({"tool": name, "usage": result.get("usage"), "actual_fee": result.get("actual_fee")})
        if action:
            entry = b["cost_ledger"][-1]
            entry["actual"] = result.get("cost")
            actual = result.get("cost") or {}
            # Settle comparable simulated units; monetary invoices stay separate
            # from local unknown-fee reservations and are retained verbatim.
            declared_unit = (entry.get("declared_cost") or {}).get("unit")
            if (actual.get("status") == "simulated" and entry["account"] == "simulated_cost"
                    and (not declared_unit or not actual.get("unit") or actual.get("unit") == declared_unit)):
                settled, _ = self._action_cost({"cost": actual})
                b["simulated_cost"] += settled - entry["amount"]
                entry["settled_amount"] = settled
        run["pending_call"].update(phase="RETURNED_NOT_APPLIED", output_ref=output_ref)
        self._event(run, "tool_returned", {"tool": name, "request_id": request_id, "input_ref": input_ref,
                    "output_ref": output_ref, "elapsed_seconds": time.perf_counter() - started})
        return result

    def _finish(self, run, state, reason, detail=None):
        run["state"], run["reason"] = state, reason
        run["stop_audit"] = self._stop_audit(run, state, reason, detail)
        self._event(run, "stopped" if state == "STOPPED" else "deferred",
                    {"reason": reason, "detail": detail, "stop_audit": deepcopy(run["stop_audit"])})

    def _event(self, run, kind, data):
        if kind not in {"tool_started", "tool_returned"} and (run.get("pending_call") or {}).get("phase") == "RETURNED_NOT_APPLIED":
            run["pending_call"] = None
        snapshot = deepcopy({k: v for k, v in run.items() if k != "events"})
        event = redact({"seq": len(run["events"]) + 1, "at": now(), "type": kind, "data": data, "snapshot": snapshot,
                        "previous_hash": run["events"][-1]["event_hash"] if run["events"] else None})
        event["event_hash"] = digest(event)
        run["events"].append(event)
        self.store.save(run)
