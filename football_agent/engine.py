"""Deterministic, incrementally executable tool workflow with durable auditing."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import math
from queue import Empty, Queue
from threading import RLock, Thread
import time
from uuid import uuid4

from .audit import RunStore, digest, redact
from .policy import FixedOrderPolicy
from .types import Budget, Decision, ToolResult, now, utc


TERMINAL = {"STOPPED", "DEFERRED"}
PROBABILITIES = ("home_win_probability", "draw_probability", "away_win_probability")
CONFIGURATION_IDENTITY = ("mode", "archive_root", "refresh_mechanism", "scenario", "model_artifact", "model_id", "model_version",
                          "artifact_sha256", "trained_at", "information_mode", "time_mode", "export_path", "export_sha256",
                          "reader_version", "configuration_hash")


class Workflow:
    def __init__(self, tools, store_dir="var/parallel/agent/runs", policy=None):
        self.tools = tools
        self.store = RunStore(store_dir)
        self.policy = policy or FixedOrderPolicy()
        self._lock = RLock()

    def capabilities(self):
        return {"mode": self.tools.mode, "policy": self.policy.version,
                "get_run": "connected_local_journal", **self.tools.capabilities()}

    def list_matches(self, cutoff=None):
        inputs = {} if cutoff is None else {"prediction_cutoff": utc(cutoff).isoformat()}
        try:
            return self.tools.execute("query_matches", inputs)
        except Exception as exc:
            return ToolResult("FAILED", error={"category": type(exc).__name__})

    def start(self, match_id, prediction_cutoff, budget=None, *, _run_id=None, _lineage=None):
        with self._lock:
            run = self._initial_run(match_id, prediction_cutoff, budget, _run_id or str(uuid4()),
                                    self.tools.capabilities(), self.policy.version,
                                    getattr(self.policy, "configuration", {}), _lineage)
            with self.store.lock(run["run_id"]):
                if (self.store._folder(run["run_id"]) / "run.json").exists():
                    raise FileExistsError("A run with this ID already exists; its audit journal is immutable")
                self._initialize(run)
                return self._result(run)

    def _initial_run(self, match_id, prediction_cutoff, budget, run_id, configuration,
                     policy_version, policy_configuration, lineage=None):
        """Build a frozen initial snapshot without executing tools or saving it."""
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError("match_id must be nonempty")
        cutoff = utc(prediction_cutoff).isoformat()
        limits = budget if isinstance(budget, Budget) else Budget(**(budget or {}))
        configuration = deepcopy(configuration)
        mode = configuration.get("mode", self.tools.mode)
        run = {"schema_version": "agent-run-v2", "run_id": run_id,
                   "match_id": match_id, "prediction_cutoff": cutoff,
                   "mode": mode, "test_mode": mode == "test",
                   "information_mode": configuration.get("information_mode", "synthetic_test" if mode == "test" else "operational"), "policy_version": policy_version,
                   "time_mode": configuration.get("time_mode", "archive_replay"),
                   "tool_configuration": configuration,
                   "policy_configuration": deepcopy(policy_configuration),
                   "state": "INITIALIZED", "reason": "RUN_INITIALIZED", "created_at": now(),
                   "match": None, "evidence": [], "features": None, "prediction": None,
                   "last_prediction": None, "queried_topics": [], "quality": [],
                   "budget": {**asdict(limits), "calls": 0, "queries": 0,
                              "spent_cost": 0.0, "elapsed_ms": 0.0, "cost_unit": "credits",
                              "cost_basis": (configuration.get("service") or {}).get("cost_basis", "adapter_reported_local_io_zero"),
                              "estimated_cost_charges": 0, "requests": 0},
                   "events": [], "pending_call": None, "transition_in_progress": None,
                   "unreconciled_calls": [], "resume_state": None, "retry_at": None,
                   "refresh_results": []}
        if lineage:
            run.update(deepcopy(lineage))
        self._remaining(run)
        return run

    def _initialize(self, run):
        self._event(run, "initialized", {"capabilities": {"mode": run["mode"], "policy": run["policy_version"],
                    "get_run": "connected_local_journal", **run["tool_configuration"]}})

    def _configuration_matches(self, run):
        configured, original = self.tools.capabilities(), run.get("tool_configuration", {})
        return (all(original.get(k) == configured.get(k) for k in CONFIGURATION_IDENTITY)
                and run["policy_version"] == self.policy.version
                and run.get("policy_configuration", {}) == getattr(self.policy, "configuration", {}))

    def get_run(self, run_id):
        with self._lock, self.store.lock(run_id):
            run = self.store.load(run_id)
            # A crash between tool intent and result is not silently retried.
            if run.get("pending_call") or run.get("transition_in_progress"):
                self._recover(run)
            if run["state"] not in TERMINAL:
                self._ensure_parent_committed(run)
            return self._result(run)

    def run_to_completion(self, run_id):
        with self._lock, self.store.lock(run_id):
            while True:
                run = self.get_run(run_id)
                if run["state"] in TERMINAL | {"WAITING"}:
                    return run
                self.step(run_id)

    def step(self, run_id, action=None, topic=None):
        with self._lock, self.store.lock(run_id):
            run = self.store.load(run_id)
            if run["mode"] != self.tools.mode:
                raise ValueError("Run mode differs from configured tools")
            if run.get("pending_call") or run.get("transition_in_progress"):
                self._recover(run)
                return self._result(run)
            if run["state"] in TERMINAL:
                return self._result(run)
            if not self._ensure_parent_committed(run):
                return self._result(run)
            if not self._configuration_matches(run):
                self._finish(run, "DEFERRED", "RUN_CONFIGURATION_CHANGED")
                return self._result(run)
            if action not in {None, "next", "query", "wait", "resume", "stop", "defer"}:
                raise ValueError("Unknown workflow action")
            if run["state"] == "WAITING" and action not in {"stop", "defer", "resume"}:
                return self._result(run)
            if run["state"] == "WAITING" and action == "resume" and run.get("retry_at") and datetime.now(timezone.utc) < utc(run["retry_at"]):
                return self._result(run)
            if action == "query" and (run["state"] != "PREDICTED" or not isinstance(topic, str) or not topic.strip() or len(topic) > 100):
                raise ValueError("A predicted run and nonempty topic are required for query")
            run["transition_in_progress"] = {"id": str(uuid4()), "state": run["state"], "action": action or "next"}
            self._event(run, "transition_started", run["transition_in_progress"])
            if action in {"stop", "defer"}:
                self._finish(run, "STOPPED" if action == "stop" else "DEFERRED", "USER_" + action.upper())
                return self._result(run)
            if action == "wait":
                if run["state"] != "WAITING":
                    self._wait(run, "USER_WAIT")
                return self._result(run)
            if run["state"] == "WAITING":
                if action != "resume":
                    return self._result(run)
                if run.get("retry_at") and datetime.now(timezone.utc) < utc(run["retry_at"]):
                    return self._result(run)
                run["state"] = run["resume_state"]
                run["reason"] = "USER_RESUMED"
                run["retry_at"] = None
                self._event(run, "resumed", {})
                return self._result(run)
            if action == "resume":
                return self._result(run)
            if action == "query":
                if run["state"] != "PREDICTED":
                    raise ValueError("Complete initial tools before requesting additional information")
                self._query(run, topic)
                return self._result(run)
            inputs = {"match_id": run["match_id"], "prediction_cutoff": run["prediction_cutoff"]}
            if run["state"] == "INITIALIZED":
                result = self._call(run, "query_matches", inputs)
                if self._failed(run, result):
                    return self._result(run)
                if not isinstance(result.data, list) or any(not isinstance(item, dict) for item in result.data):
                    self._finish(run, "DEFERRED", "INVALID_MATCH_RESULT")
                    return self._result(run)
                matches = result.data
                matches = [m for m in matches if str(m.get("match_id")) == run["match_id"]]
                if len(matches) != 1:
                    self._finish(run, "DEFERRED", "MATCH_NOT_FOUND" if not matches else "MATCH_CONFLICT")
                else:
                    match = matches[0]
                    try:
                        cutoff = utc(run["prediction_cutoff"])
                        if utc(match["kickoff"]) <= cutoff:
                            raise ValueError("Not prematch")
                        available = match.get("fixture_available_at", match.get("available_at"))
                        if available is None or utc(available) > cutoff:
                            raise ValueError("Fixture unavailable")
                        if match.get("status", "scheduled") not in {"scheduled", "unknown"}:
                            raise ValueError("Not scheduled")
                    except (ValueError, KeyError, TypeError):
                        self._finish(run, "DEFERRED", "MATCH_NOT_VISIBLE_OR_NOT_PREMATCH")
                    else:
                        run["match"] = match
                        run["state"], run["reason"] = "MATCH_READY", "MATCH_VISIBLE_AT_CUTOFF"
                        self._event(run, "match_selected", {"match": match})
            elif run["state"] == "MATCH_READY":
                result = self._call(run, "read_evidence", inputs)
                if self._failed(run, result):
                    return self._result(run)
                added = self._ingest(run, result.data)
                if run["state"] not in TERMINAL:
                    if not added:
                        self._finish(run, "DEFERRED", "NO_VALID_EVIDENCE")
                    else:
                        run["state"], run["reason"] = "EVIDENCE_READY", "EVIDENCE_AVAILABLE"
                        self._event(run, "evidence_ready", {"new_evidence": added})
            elif run["state"] == "EVIDENCE_READY":
                result = self._call(run, "get_features", {**inputs, "evidence": run["evidence"], "match": run["match"]})
                if self._failed(run, result):
                    return self._result(run)
                features = result.data
                try:
                    if not isinstance(features, dict) or not features.get("feature_version"):
                        raise ValueError("Missing feature version")
                    if str(features.get("match_id", run["match_id"])) != run["match_id"]:
                        raise ValueError("Feature match mismatch")
                    if utc(features.get("prediction_cutoff", run["prediction_cutoff"])) != utc(run["prediction_cutoff"]):
                        raise ValueError("Feature cutoff mismatch")
                    if features.get("information_mode", run["information_mode"]) != run["information_mode"]:
                        raise ValueError("Feature mode mismatch")
                    if not features.get("input_manifest"):
                        raise ValueError("Missing feature lineage")
                    if features.get("features_available_at") and utc(features["features_available_at"]) > utc(run["prediction_cutoff"]):
                        raise ValueError("Features unavailable at cutoff")
                    expected_version = run["tool_configuration"].get("expected_feature_version")
                    if expected_version and features["feature_version"] != expected_version:
                        raise ValueError("Unexpected platform feature version")
                except (ValueError, TypeError):
                    self._finish(run, "DEFERRED", "INVALID_FEATURE_RESULT")
                else:
                    run["features"] = features
                    run["state"], run["reason"] = "FEATURES_READY", "FEATURES_COMPUTED"
                    self._event(run, "features_ready", {"feature_version": features["feature_version"]})
            elif run["state"] == "FEATURES_READY":
                result = self._call(run, "predict", {**inputs, "evidence": run["evidence"],
                                    "features": run["features"], "match": run["match"]})
                if self._failed(run, result):
                    return self._result(run)
                if not self._valid_prediction(run, result.data):
                    self._finish(run, "DEFERRED", "INVALID_MODEL_RESULT")
                else:
                    before = run["last_prediction"]
                    run["prediction"] = result.data
                    run["state"], run["reason"] = "PREDICTED", "MODEL_RETURNED_PROBABILITIES"
                    self._event(run, "prediction_updated", {"before": before, "after": result.data,
                                "evidence_versions": self._versions(run), "feature_version": run["features"]["feature_version"],
                                "update_reason": "INITIAL_PREDICTION" if before is None else "NEW_VALID_EVIDENCE"})
            elif run["state"] == "PREDICTED":
                try:
                    if getattr(self.policy, "tool_name", None):
                        planned = self._call(run, self.policy.tool_name, {"run": self._snapshot(run)})
                        if self._failed(run, planned):
                            return self._result(run)
                        decision = self.policy.decode(planned.data)
                    else:
                        decision = self.policy.decide(deepcopy(self._snapshot(run)))
                    if not isinstance(decision, Decision) or decision.action not in {"query", "wait", "stop", "defer"}:
                        raise ValueError("Invalid policy result")
                    if decision.action == "query" and (not isinstance(decision.topic, str) or not decision.topic.strip() or len(decision.topic) > 100):
                        raise ValueError("Invalid policy topic")
                except Exception as exc:
                    self._finish(run, "DEFERRED", "POLICY_ERROR_" + type(exc).__name__)
                else:
                    self._event(run, "policy_decision", asdict(decision))
                    if decision.action == "query":
                        self._query(run, decision.topic)
                    elif decision.action == "wait":
                        self._wait(run, decision.reason or "POLICY_WAIT")
                    else:
                        self._finish(run, "STOPPED" if decision.action == "stop" else "DEFERRED",
                                     decision.reason or "POLICY_" + decision.action.upper())
            else:
                self._finish(run, "DEFERRED", "INVALID_RUN_STATE")
            return self._result(run)

    def _query(self, run, topic):
        if not isinstance(topic, str) or not topic.strip() or len(topic) > 100:
            raise ValueError("Specify a nonempty information topic")
        if topic in run["queried_topics"]:
            self._finish(run, "STOPPED", "REPEATED_TOPIC_WITHOUT_NEW_EVIDENCE")
            return
        result = self._call(run, "refresh_information", {"match_id": run["match_id"],
                            "prediction_cutoff": run["prediction_cutoff"], "topic": topic})
        if result is None:
            return
        # RATE_LIMITED/timeout require explicit handling; no loop retries.
        if result.status == "RATE_LIMITED":
            self._wait(run, "RATE_LIMITED", result.retry_after_seconds)
            return
        run["queried_topics"].append(topic)
        if result.status == "NO_DATA":
            self._finish(run, "STOPPED" if run["prediction"] else "DEFERRED", "NO_NEW_VALID_EVIDENCE")
            return
        if self._failed(run, result):
            return
        data = result.data
        if run.get("time_mode") == "live":
            self._advance_live(run, result)
            return
        added = self._ingest(run, data.get("evidence", []) if isinstance(data, dict) else data)
        if run["state"] in TERMINAL:
            return
        if not added:
            self._finish(run, "STOPPED" if run["prediction"] else "DEFERRED", "NO_NEW_VALID_EVIDENCE")
            return
        run["last_prediction"] = run["prediction"] or run["last_prediction"]
        run["prediction"] = None
        run["features"] = None
        run["state"], run["reason"] = "EVIDENCE_READY", "NEW_VALID_EVIDENCE"
        self._event(run, "evidence_ready", {"new_evidence": added, "topic": topic})

    def _advance_live(self, run, result):
        data = result.data if isinstance(result.data, dict) else {}
        try:
            cutoff = utc(data["decision_cutoff"])
            if cutoff <= utc(run["prediction_cutoff"]) or cutoff >= utc(run["match"]["kickoff"]):
                raise ValueError("Live decision must advance before kickoff")
            if cutoff > datetime.now(timezone.utc) + timedelta(seconds=2):
                raise ValueError("Live decision is in the future")
            refs = data.get("observation_refs")
            if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
                raise ValueError("Live refresh requires actual observation references")
        except (KeyError, TypeError, ValueError):
            self._finish(run, "DEFERRED", "INVALID_LIVE_REFRESH_RESULT")
            return
        if not self._configuration_matches(run):
            self._finish(run, "DEFERRED", "RUN_CONFIGURATION_CHANGED_DURING_REFRESH")
            return
        child_id = str(uuid4())
        lineage = {"parent_run_id": run["run_id"], "root_run_id": run.get("root_run_id", run["run_id"]),
                   "budget": deepcopy(run["budget"]), "inherited_budget": deepcopy(run["budget"]),
                   "queried_topics": list(run["queried_topics"]), "refresh_observation_refs": data["observation_refs"],
                   "mode": run["mode"], "information_mode": run["information_mode"], "time_mode": run["time_mode"],
                   "previous_decision": {"run_id": run["run_id"], "prediction_cutoff": run["prediction_cutoff"],
                                         "prediction": run["prediction"], "evidence_versions": self._versions(run)}}
        limits = Budget(**{k: run["budget"][k] for k in Budget.__dataclass_fields__})
        child = self._initial_run(run["match_id"], cutoff, limits, child_id, run["tool_configuration"],
                                  run["policy_version"], run["policy_configuration"], lineage)
        # One durable intent freezes the child identity/configuration/budget.
        # Recovery reuses it verbatim and never repeats the successful refresh.
        run["child_run_id"] = child_id
        run["live_handoff"] = {"status": "pending", "child_run_id": child_id,
                               "child_snapshot": child, "child_snapshot_hash": digest(self._snapshot(child))}
        self._event(run, "live_refresh_archived", {"child_run_id": child_id,
                    "next_prediction_cutoff": cutoff.isoformat(), "observation_refs": data["observation_refs"],
                    "content_changed": data.get("content_changed"), "refresh": data.get("refresh"),
                    "child_snapshot_hash": run["live_handoff"]["child_snapshot_hash"]})
        self._complete_live_handoff(run)

    def _complete_live_handoff(self, run, *, recovering=False):
        """Finish a durable parent/child transfer without any tool invocation.

        The caller owns the parent lock. Child locks are nonblocking, including
        when a child reader already owns its lock and asks us to recover the
        parent on the same thread. Contention therefore returns RUN_BUSY rather
        than waiting in an inverted parent/child lock order.
        """
        intent = run["live_handoff"]
        template = intent["child_snapshot"]
        child_id = intent["child_run_id"]
        expected_hash = intent["child_snapshot_hash"]
        if (run.get("child_run_id") != child_id or template.get("run_id") != child_id
                or template.get("parent_run_id") != run["run_id"]
                or digest(self._snapshot(template)) != expected_hash):
            raise ValueError("Live handoff snapshot identity/checksum mismatch")
        with self.store.lock(child_id):
            try:
                child = self.store.load(child_id)
            except FileNotFoundError:
                child = deepcopy(template)
                self._initialize(child)
            else:
                # Verify genesis, even if the child already has later events.
                # Never overwrite an existing journal or reset its counters.
                if digest(child["events"][0]["snapshot"]) != expected_hash:
                    raise ValueError("Existing child differs from the committed handoff snapshot")
            intent["status"] = "committed"
            if recovering:
                self._event(run, "live_handoff_recovered", {"child_run_id": child_id,
                            "child_snapshot_hash": expected_hash, "repeated_tool_calls": 0})
            self._finish(run, "STOPPED", "LIVE_CUTOFF_ADVANCED")

    def _ensure_parent_committed(self, run):
        parent_id = run.get("parent_run_id")
        if not parent_id:
            return True
        if parent_id == run["run_id"]:
            self._finish(run, "DEFERRED", "INVALID_PARENT_RUN_IDENTITY")
            return False
        with self.store.lock(parent_id):
            try:
                parent = self.store.load(parent_id)
            except FileNotFoundError:
                self._finish(run, "DEFERRED", "PARENT_RUN_MISSING")
                return False
            if parent.get("pending_call") or parent.get("transition_in_progress"):
                self._recover(parent)
            if parent.get("child_run_id") != run["run_id"] or parent.get("reason") != "LIVE_CUTOFF_ADVANCED":
                self._finish(run, "DEFERRED", "PARENT_BUDGET_TRANSFER_NOT_COMMITTED")
                return False
            # Parent is now terminal. Its frozen transfer budget/configuration
            # must be the child's genesis, even after cumulative child spending.
            if (any(run.get(key) != parent.get(key) for key in ("match_id", "mode", "information_mode", "time_mode",
                                                               "tool_configuration", "policy_version", "policy_configuration"))
                    or utc(run["prediction_cutoff"]) <= utc(parent["prediction_cutoff"])
                    or run.get("root_run_id") != parent.get("root_run_id", parent_id)
                    or run.get("inherited_budget") != parent["budget"]
                    or any(run["budget"].get(key) != parent["budget"].get(key) for key in Budget.__dataclass_fields__)
                    or any(run["budget"].get(key, 0) < parent["budget"].get(key, 0)
                           for key in ("calls", "queries", "requests", "spent_cost", "elapsed_ms"))):
                self._finish(run, "DEFERRED", "PARENT_RUN_IDENTITY_OR_BUDGET_MISMATCH")
                return False
            return True

    def _call(self, run, tool, inputs):
        budget = run["budget"]
        inputs = deepcopy(inputs)
        inputs.update(information_mode=run["information_mode"], time_mode=run.get("time_mode", "archive_replay"))
        try:
            estimate = self.tools.estimate_cost(tool, deepcopy(inputs))
            if isinstance(estimate, bool) or not math.isfinite(estimate) or estimate < 0:
                raise ValueError("Invalid cost estimate")
            estimate_requests = getattr(self.tools, "estimate_requests", lambda *_: 0)(tool, deepcopy(inputs))
            if type(estimate_requests) is not int or estimate_requests < 0:
                raise ValueError("Invalid request estimate")
        except Exception as exc:
            self._finish(run, "DEFERRED", "COST_ESTIMATE_ERROR_" + type(exc).__name__)
            return None
        self._remaining(run)
        reason = None
        if budget["remaining_calls"] <= 0:
            reason = "CALL_BUDGET_EXHAUSTED"
        elif tool == "refresh_information" and budget["remaining_queries"] <= 0:
            reason = "QUERY_BUDGET_EXHAUSTED"
        elif estimate > budget["remaining_cost"] + 1e-12:
            reason = "COST_BUDGET_EXHAUSTED"
        elif estimate_requests > budget["remaining_requests"]:
            reason = "REQUEST_BUDGET_EXHAUSTED"
        elif budget["remaining_elapsed_ms"] <= 0:
            reason = "LATENCY_BUDGET_EXHAUSTED"
        if reason:
            self._finish(run, "STOPPED" if run["prediction"] else "DEFERRED", reason)
            return None
        call_id = str(uuid4())
        if tool == "refresh_information":
            inputs.update(request_id=call_id, max_requests=min(budget["remaining_requests"], estimate_requests) if estimate_requests else budget["remaining_requests"],
                          timeout_seconds=min(budget["max_tool_seconds"], budget["remaining_elapsed_ms"] / 1000))
        input_ref = self.store.put_blob(run["run_id"], inputs)
        budget["calls"] += 1
        budget["queries"] += int(tool == "refresh_information")
        run["pending_call"] = {"call_id": call_id, "tool": tool, "estimated_cost": estimate, "estimated_requests": estimate_requests}
        self._event(run, "tool_started", {"call_id": call_id, "tool": tool, "input_ref": input_ref,
                                          "estimated_cost": estimate, "estimated_requests": estimate_requests})
        started = time.perf_counter()
        timeout = min(budget["max_tool_seconds"], budget["remaining_elapsed_ms"] / 1000)
        queue = Queue(maxsize=1)

        def invoke():
            try:
                queue.put((self.tools.execute(tool, deepcopy(inputs)), None))
            except Exception as exc:
                queue.put((None, type(exc).__name__))

        worker = Thread(target=invoke, daemon=True, name="agent-tool-" + call_id)
        worker.start()
        actual_known = True
        try:
            result, error = queue.get(timeout=timeout)
            if error:
                actual_known = False
                result = ToolResult("TIMEOUT" if error == "TimeoutError" else "FAILED", error={"category": error}, cost=estimate, requests=estimate_requests)
        except Empty:
            actual_known = False
            result = ToolResult("TIMEOUT", error={"category": "ExecutionDeadline",
                               "late_result_discarded": True, "external_effects_require_reconciliation": True}, cost=estimate, requests=estimate_requests)
        elapsed = (time.perf_counter() - started) * 1000
        if not isinstance(result, ToolResult):
            result = ToolResult("FAILED", error={"category": "InvalidToolResult"}, cost=estimate, requests=estimate_requests)
            actual_known = False
        if not result.cost_known or not result.requests_known:
            actual_known = False
        if isinstance(result.cost, bool) or not isinstance(result.cost, (int, float)) or not math.isfinite(result.cost) or result.cost < 0:
            result = ToolResult("FAILED", error={"category": "InvalidReportedCost"}, cost=estimate, requests=estimate_requests)
            actual_known = False
        if type(result.requests) is not int or result.requests < 0:
            result = ToolResult("FAILED", error={"category": "InvalidReportedRequests"}, cost=result.cost, requests=estimate_requests)
            actual_known = False
        try:
            output_ref = self.store.put_blob(run["run_id"], asdict(result))
        except (ValueError, TypeError):
            result = ToolResult("FAILED", error={"category": "NonJsonToolResult"}, cost=result.cost, requests=result.requests)
            output_ref = self.store.put_blob(run["run_id"], asdict(result))
        budget["spent_cost"] = round(budget["spent_cost"] + result.cost, 12)
        budget["elapsed_ms"] += elapsed
        budget["estimated_cost_charges"] += int(not actual_known)
        budget["requests"] = budget.get("requests", 0) + result.requests
        run["pending_call"] = None
        if tool == "refresh_information":
            data = result.data if isinstance(result.data, dict) else {}
            metadata = {key: deepcopy(value) for key, value in data.items()
                        if key not in {"evidence", "features", "raw_response", "body", "records"}}
            run.setdefault("refresh_results", []).append({"call_id": call_id, "status": str(result.status),
                "error": result.error, "cost": result.cost, "requests": result.requests, "latency_ms": elapsed,
                "actual_cost_known": actual_known, "actual_requests_known": actual_known,
                "input_ref": input_ref, "output_ref": output_ref, "observation_refs": data.get("observation_refs", []),
                "decision_cutoff": data.get("decision_cutoff"), "content_changed": data.get("content_changed"),
                "metadata": metadata})
        self._event(run, "tool_returned", {"call_id": call_id, "tool": tool,
                    "input_ref": input_ref, "output_ref": output_ref, "status": str(result.status),
                    "error": result.error, "cost": result.cost, "actual_cost_known": actual_known,
                    "latency_ms": elapsed, "retry_after_seconds": result.retry_after_seconds,
                    "requests": result.requests, "actual_requests_known": actual_known})
        if budget["requests"] > budget["max_requests"]:
            self._finish(run, "DEFERRED", "TOOL_REPORTED_REQUESTS_EXCEEDED_BUDGET")
            return None
        if budget["spent_cost"] > budget["max_cost"] + 1e-12:
            self._finish(run, "DEFERRED", "TOOL_REPORTED_COST_EXCEEDED_BUDGET")
            return None
        if budget["elapsed_ms"] >= budget["max_elapsed_seconds"] * 1000:
            self._finish(run, "DEFERRED", "LATENCY_BUDGET_EXHAUSTED")
            return None
        return result

    def _failed(self, run, result):
        if result is None:
            return True
        if result.status == "OK":
            return False
        if result.status == "RATE_LIMITED":
            self._wait(run, "RATE_LIMITED", result.retry_after_seconds)
        else:
            self._finish(run, "DEFERRED", str(result.status))
        return True

    def _ingest(self, run, items):
        if not isinstance(items, list):
            self._finish(run, "DEFERRED", "INVALID_EVIDENCE_RESULT")
            return []
        cutoff = utc(run["prediction_cutoff"])
        added = []
        for item in items:
            reason = None
            try:
                if not isinstance(item, dict) or not all(item.get(k) for k in
                        ("evidence_id", "version", "source", "kind", "content_hash", "raw_ref")):
                    raise ValueError("Missing evidence provenance")
                if not isinstance(item.get("content"), dict):
                    raise ValueError("Invalid evidence body")
                available, observed = utc(item["available_at"]), utc(item["observed_at"])
                if item.get("match_id") is not None and str(item["match_id"]) != run["match_id"]:
                    reason = "OTHER_MATCH"
                elif item.get("information_mode", run["information_mode"]) != run["information_mode"]:
                    reason = "INFORMATION_MODE_MISMATCH"
                elif run["information_mode"] == "historical_reconstruction":
                    assumptions = item.get("reconstruction") or run["tool_configuration"].get("reconstruction_assumptions")
                    if not isinstance(assumptions, dict) or not assumptions.get("basis") or not assumptions.get("revision_risk"):
                        reason = "MISSING_RECONSTRUCTION_ASSUMPTIONS"
                    elif available > cutoff:
                        reason = "OUTSIDE_CUTOFF"
                elif observed > cutoff or available > cutoff or available < observed:
                    reason = "OUTSIDE_CUTOFF"
                if reason is None and item.get("source_published_at") and utc(item["source_published_at"]) > cutoff:
                    reason = "OUTSIDE_CUTOFF"
                if reason is None and item.get("expires_at") and utc(item["expires_at"]) <= cutoff:
                    reason = "STALE_EVIDENCE"
            except (ValueError, TypeError, KeyError):
                reason = "INVALID_EVIDENCE"
            if reason:
                warning = {"evidence_id": item.get("evidence_id") if isinstance(item, dict) else None, "reason": reason}
                run["quality"].append(warning)
                self._event(run, "evidence_rejected", warning)
                continue
            duplicate = next((e for e in run["evidence"] if (e["source"], e["kind"], e["content_hash"]) ==
                              (item["source"], item["kind"], item["content_hash"])), None)
            if duplicate:
                if item.get("conflict"):
                    run["quality"].append({"evidence_id": item["evidence_id"], "reason": "CONFLICTING_EVIDENCE"})
                    run["last_prediction"] = run["prediction"] or run["last_prediction"]
                    run["prediction"] = None
                    self._event(run, "evidence_conflict", {"incoming": item, "conflicting_ids": [duplicate["evidence_id"]]})
                    self._finish(run, "DEFERRED", "CONFLICTING_EVIDENCE")
                    return []
                self._event(run, "evidence_duplicate", {"evidence_id": item["evidence_id"],
                            "retained_evidence_id": duplicate["evidence_id"], "source": item["source"]})
                continue
            # Same source can revise a fact. Equal timestamps with different
            # content or disagreement across sources requires upstream resolution.
            related = [e for e in run["evidence"] if
                       (item.get("conflict_key") and e.get("conflict_key") == item["conflict_key"])
                       or (e["source"] == item["source"] and e["evidence_id"] == item["evidence_id"])]
            conflicts = [e for e in related if e["content_hash"] != item["content_hash"] and
                         (e["source"] != item["source"] or utc(e["available_at"]) == available)]
            if conflicts or item.get("conflict"):
                run["quality"].append({"evidence_id": item["evidence_id"], "reason": "CONFLICTING_EVIDENCE"})
                run["last_prediction"] = run["prediction"] or run["last_prediction"]
                run["prediction"] = None
                self._event(run, "evidence_conflict", {"incoming": item, "conflicting_ids": [e["evidence_id"] for e in conflicts]})
                self._finish(run, "DEFERRED", "CONFLICTING_EVIDENCE")
                return []
            if any(utc(e["available_at"]) > available for e in related):
                self._event(run, "evidence_rejected", {"evidence_id": item["evidence_id"], "reason": "SUPERSEDED_VERSION"})
                continue
            for old in related:
                run["evidence"].remove(old)
            if run["prediction"] is not None:
                run["last_prediction"], run["prediction"] = run["prediction"], None
            run["evidence"].append(deepcopy(item))
            added.append(item["evidence_id"])
            self._event(run, "evidence_arrived", {"evidence": item, "superseded_ids": [e["evidence_id"] for e in related]})
        return added

    @staticmethod
    def _valid_prediction(run, prediction):
        try:
            if str(prediction["match_id"]) != run["match_id"] or utc(prediction["prediction_cutoff"]) != utc(run["prediction_cutoff"]):
                return False
            if run["mode"] == "real" and prediction.get("test_mode"):
                return False
            expected_model = run["tool_configuration"].get("model_version")
            if expected_model and prediction.get("model_version") != expected_model:
                return False
            values = [prediction[key] for key in PROBABILITIES]
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
                return False
            return (math.isclose(sum(values), 1.0, abs_tol=1e-9) and bool(prediction.get("model_version"))
                    and bool(prediction.get("dataset_version")) and bool(prediction.get("input_manifest"))
                    and prediction.get("information_mode") == run["information_mode"])
        except (KeyError, TypeError, ValueError):
            return False

    def _wait(self, run, reason, retry_after=None):
        run["resume_state"] = run["state"]
        run["state"], run["reason"] = "WAITING", reason
        if retry_after is not None and isinstance(retry_after, (int, float)) and math.isfinite(retry_after):
            run["retry_at"] = (datetime.now(timezone.utc) + timedelta(seconds=max(0, retry_after))).isoformat()
        self._event(run, "waiting", {"reason": reason, "retry_at": run["retry_at"]})

    def _finish(self, run, state, reason):
        run["state"], run["reason"] = state, reason
        run["transition_in_progress"] = None
        self._event(run, "stopped" if state == "STOPPED" else "deferred", {"reason": reason})

    def _recover(self, run):
        if run.get("live_handoff") and not run.get("pending_call"):
            self._complete_live_handoff(run, recovering=True)
            return
        pending = run.get("pending_call")
        if pending:
            reservation = pending["estimated_cost"]
            run["budget"]["spent_cost"] = round(run["budget"]["spent_cost"] + reservation, 12)
            run["budget"]["estimated_cost_charges"] += 1
            run["budget"]["requests"] = run["budget"].get("requests", 0) + pending.get("estimated_requests", 0)
            run.setdefault("unreconciled_calls", []).append({**pending, "cost_status": "UNKNOWN_RESERVED_ESTIMATE",
                                                            "reserved_cost": reservation})
        interrupted = {"pending_call": pending, "transition": run.get("transition_in_progress")}
        run["pending_call"] = None
        run["transition_in_progress"] = None
        run["state"] = "DEFERRED"
        run["reason"] = "INTERRUPTED_TOOL_CALL_REQUIRES_RECONCILIATION" if pending else "INTERRUPTED_TRANSITION_REQUIRES_REVIEW"
        self._event(run, "interrupted_transition", interrupted)
        self._finish(run, "DEFERRED", run["reason"])

    @staticmethod
    def _remaining(run):
        b = run["budget"]
        b["remaining_calls"] = max(0, b["max_calls"] - b["calls"])
        b["remaining_queries"] = max(0, b["max_queries"] - b["queries"])
        b["remaining_cost"] = max(0.0, round(b["max_cost"] - b["spent_cost"], 12))
        b["remaining_elapsed_ms"] = max(0.0, b["max_elapsed_seconds"] * 1000 - b["elapsed_ms"])
        b.setdefault("max_requests", 6)
        b.setdefault("requests", 0)
        b["remaining_requests"] = max(0, b["max_requests"] - b["requests"])

    @staticmethod
    def _versions(run):
        return [{k: e[k] for k in ("evidence_id", "version", "source", "content_hash", "raw_ref")}
                for e in run["evidence"]]

    @staticmethod
    def _snapshot(run):
        return deepcopy({k: v for k, v in run.items() if k not in {"events", "report"}})

    def _event(self, run, kind, data):
        self._remaining(run)
        event = redact({"seq": len(run["events"]) + 1, "type": kind, "at": now(), "data": data,
                        "snapshot": self._snapshot(run),
                        "previous_hash": run["events"][-1]["event_hash"] if run["events"] else None})
        event["event_hash"] = digest(event)
        run["events"].append(event)
        self.store.save(run)

    def _result(self, run):
        if run.get("transition_in_progress") and not run.get("pending_call"):
            run["transition_in_progress"] = None
            self._event(run, "transition_completed", {})
        result = deepcopy(run)
        prediction = run["prediction"]
        result["report"] = {"run_id": run["run_id"], "match_id": run["match_id"],
                            "prediction_cutoff": run["prediction_cutoff"],
                            "information_mode": run["information_mode"], "time_mode": run.get("time_mode", "archive_replay"),
                            "model_configuration": run.get("tool_configuration", {}),
                            "parent_run_id": run.get("parent_run_id"), "child_run_id": run.get("child_run_id"),
                            "label": ("测试模式 · 合成压力测试" if run["test_mode"] else
                                      "真实历史重建 · 明示可用时间假设" if run["information_mode"] == "historical_reconstruction" else
                                      "真实在线决策 · 关联时点运行" if run.get("time_mode") == "live" else "真实档案回放"),
                            "state": run["state"], "reason": run["reason"], "prediction": prediction,
                            "evidence_versions": self._versions(run), "quality": run["quality"],
                            "budget": run["budget"], "unreconciled_calls": run.get("unreconciled_calls", []),
                            "refresh_results": run.get("refresh_results", []),
                            "refresh_observation_refs": run.get("refresh_observation_refs", []),
                            "model_version": prediction.get("model_version") if prediction else None,
                            "feature_version": (run["features"] or {}).get("feature_version"),
                            "claims": [{"evidence_id": e["evidence_id"], "source": e["source"],
                                        "observed_at": e["observed_at"], "content": e["content"]} for e in run["evidence"]],
                            "interpretation": "概率由预测工具返回；证据为工具输入，未推断单条信息的因果影响。",
                            "prediction_current": prediction is not None and run["state"] not in {"DEFERRED", "EVIDENCE_READY", "FEATURES_READY"}}
        return redact(result)
