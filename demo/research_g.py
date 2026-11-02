"""Executable, offline research f/g workflow for the submission UI.

This is a new run over reconstructed archives, not a replay of a live g policy.
The selector receives only a current public menu and observed evidence. Archived
successors and labels are never passed to it. f is recomputed after each reveal.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from football_research.acquisition import entity_evidence_updates as predictor
from football_research.acquisition import state_candidate_gain as immediate
from football_research.acquisition import policy_return_continuation_g as continuation
from football_research.acquisition import scoped_reviewed_f


SCHEMA = "footballmind-research-g-run-v1"
ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"rg-[a-f0-9]{32}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def observed_snapshot(row):
    """Only evidence already revealed in the current state may become g input."""
    _, audit = predictor.resolve(row)
    retained = {e["event_id"] for e in audit["events"] if e["status"] == "retained"}
    claims = []
    for event in row["evidence_events"]:
        if event["input_id"] not in row["used_input_ids"]:
            raise ValueError("Evidence is not in current revealed inputs")
        if event["event_id"] not in retained:
            continue
        facet, value, pending, _ = predictor.status(event)
        claims.append({"input_id": event["input_id"], "channel": event["evidence"]["channel"],
                       "entity": event["entity"], "team": event["evidence"]["team"],
                       "facet": facet, "value": value or "", "pending": pending,
                       "text": event["evidence"]["text"]})
    return {"physical_match_id": row["physical_match_id"],
            "used_input_ids": list(row["used_input_ids"]), "claims": claims}


class ResearchGApplication:
    """Server-owned artifact configuration; clients cannot supply paths or scores."""

    def __init__(self, store_dir, root=ROOT):
        self.root = Path(root).resolve()
        self.store_dir = Path(store_dir).resolve()
        reviews = self.root / "var/reviews"
        source = reviews / "fm-historical-transition-g-training-20261014-v2"
        gdir = reviews / "fm-g-policy-return-continuation-20261014-v1"
        self.files = {
            "f": source / "f-labeler.json", "g": gdir / "g-policy-return.json",
            "states": source / "states.json", "transitions": source / "transitions.json",
            "plan": gdir / "plan.json",
            "events": reviews / "fm-entity-updates-f-training-20261014-v1/event-training-rows.json",
            "protocol": self.root / "var/f-entity-semantic-20261011-v2/protocol.json",
        }
        protocol = read(self.files["protocol"])
        for key in ("cases", "records"):
            original = Path(protocol["source_bindings"][key]["path"])
            # Existing protocol paths belong to this repository, not client input.
            self.files[key] = original
            if digest(original) != protocol["source_bindings"][key]["sha256"]:
                raise ValueError(f"Research source binding changed: {key}")
        self.bindings = {key: digest(path) for key, path in self.files.items()}
        self.f = predictor.load(self.files["f"])
        self.g = continuation.load(self.files["g"])
        if self.g["f_labeler_sha256"] != self.bindings["f"]:
            raise ValueError("g was trained for a different f")
        self.plan = read(self.files["plan"])
        self.match_ids = self.plan["holdout_ids"]
        if set(self.match_ids) & set(self.g["fit_match_ids"]):
            raise ValueError("Demonstration cohort overlaps g fit matches")
        self.states = read(self.files["states"])
        self.graph = defaultdict(list)
        for edge in read(self.files["transitions"]):
            self.graph[edge["state_id"]].append(edge)
        for edges in self.graph.values():
            edges.sort(key=lambda e: e["action_id"])
            if len({e["action_id"] for e in edges}) != len(edges):
                raise ValueError("Duplicate action in archived menu")
        # Strip outcomes, obsolete state vectors and result clocks at the input boundary.
        self.rows = {}
        for row in read(self.files["events"]):
            key = (row["physical_match_id"], tuple(sorted(row["used_input_ids"])))
            safe = {k: deepcopy(row[k]) for k in
                    ("physical_match_id", "decision_at", "prior", "used_input_ids", "evidence_events")}
            self.rows[key] = safe
        self.fixtures = {c["fixture"]["physical_match_id"]: c["fixture"] for c in read(self.files["cases"])}
        self.records = {r["input"]["input_id"]: r["input"] for r in read(self.files["records"])}
        self.starts = {}
        for mid in self.match_ids:
            possible = [(sid, s) for sid, s in self.states.items()
                        if s["physical_match_id"] == mid and s["read_count"] == 0]
            self.starts[mid] = max(possible, key=lambda x: (x[1]["title_count"], x[0]))[0]

    def _check_bindings(self):
        if any(digest(path) != self.bindings[key] for key, path in self.files.items()):
            raise ValueError("RESEARCH_CONFIGURATION_CHANGED: restart with frozen artifacts")

    def _row(self, state_id):
        state = self.states[state_id]
        return deepcopy(self.rows[(state["physical_match_id"], tuple(sorted(state["used_input_ids"])))])

    def catalog(self):
        self._check_bindings()
        return {"schema_version": "footballmind-research-g-catalog-v1",
                "mode": "historical_research", "is_live": False, "external_requests_enabled": False,
                "status": "ready", "selection_authority": "research_g",
                "g_model_version": self.g["schema"], "f_model_version": self.f["schema"],
                "model_bindings": {k: self.bindings[k] for k in ("f", "g")},
                "production_ready": False, "calibration_status": "not_established",
                "default_max_reads": 3, "default_cost_loss_units": 0.001,
                "cost_unit": "natural_log_loss_equivalent_not_dollars",
                "strategies": ["g_two_step", "g_one_step", "fixed_order"],
                "matches": [{"match_id": mid, "home": self.fixtures[mid]["home_name"],
                             "away": self.fixtures[mid]["away_name"], "kickoff": self.fixtures[mid]["kickoff"],
                             "competition": self.fixtures[mid]["competition"],
                             "prediction_cutoff": self._row(self.starts[mid])["decision_at"],
                             "candidate_count": len(self.graph[self.starts[mid]]),
                             "eligibility": {"run_allowed": True, "scope": "historical_research_only"}}
                            for mid in self.match_ids],
                "limitations": ["历史版本与菜单可见性采用重建假设；不是真实赛前运行",
                                "复用过的开发留出集；不证明泛化优越性",
                                "使用已归档正文及审阅语义，不在运行中调用 LLM 或网络",
                                "研究 f 与近期真实比赛 shadow f 是不同工件，不能互换"]}

    def _path(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid research run ID")
        return self.store_dir / (run_id + ".json")

    def _save(self, run):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(run["run_id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load(self, run_id):
        self._check_bindings()
        run = read(self._path(run_id))
        if run["bindings"] != self.bindings:
            raise ValueError("Saved research run uses different artifacts")
        return run

    def _decision(self, run):
        remaining = run["max_reads"] - len(run["steps"])
        if remaining == 0:
            return {"action_id": None, "reason": "budget_exhausted", "candidates": []}
        edges = self.graph.get(run["state_id"], [])
        if not edges:
            return {"action_id": None, "reason": "no_feasible_candidate", "candidates": []}
        snapshot = observed_snapshot(self._row(run["state_id"]))
        menu = []
        for edge in edges:
            if edge["body_input_id"] in snapshot["used_input_ids"]:
                raise ValueError("Candidate body was already observed")
            public = deepcopy(edge["query"])
            public["probabilities"] = list(run["current"])
            public["visible_candidates"] = len(edges)
            public["acquired_documents"] = len(run["steps"])
            menu.append({"action_id": edge["action_id"], "query": immediate.build_query(public, snapshot)})
        horizon = 1 if run["strategy"] == "g_one_step" else remaining
        # No successor, result, realized gain or unread body crosses this call.
        decision = continuation.choose(self.g, deepcopy(menu), horizon, run["cost_loss_units"])
        by_id = {e["action_id"]: e for e in edges}
        for candidate in decision["candidates"]:
            edge = by_id[candidate["action_id"]]
            candidate.update(title=edge["query"]["title"], url=edge["body_url"],
                             expected_signed_gain=candidate["immediate_gain"],
                             expected_net_voi=candidate["two_step_net_value"],
                             support_status="historical_research_estimate", propensity=None)
        if run["strategy"] == "fixed_order":
            decision.update(action_id=edges[0]["action_id"], reason="fixed_order_control")
        decision["prequery_menu"] = menu
        return decision

    def start(self, payload):
        if set(payload) - {"match_id", "max_reads", "cost_loss_units", "strategy"}:
            raise ValueError("Only match_id, max_reads, cost_loss_units and strategy are accepted")
        mid = payload.get("match_id")
        if mid not in self.match_ids:
            raise ValueError("Match is not in the frozen research catalog")
        budget = payload.get("max_reads", 3)
        cost = payload.get("cost_loss_units", 0.001)
        strategy = payload.get("strategy", "g_two_step")
        if type(budget) is not int or not 1 <= budget <= 3:
            raise ValueError("max_reads must be 1..3")
        if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 0.01:
            raise ValueError("cost_loss_units must be finite within the studied range 0..0.01")
        if strategy not in ("g_two_step", "g_one_step", "fixed_order"):
            raise ValueError("Unknown research strategy")
        self._check_bindings()
        sid = self.starts[mid]
        probabilities = predictor.predict(self.f, self._row(sid)).tolist()
        run = {"schema_version": SCHEMA, "run_id": "rg-" + uuid4().hex, "match_id": mid,
               "created_at": now(), "updated_at": now(), "revision": 0, "state_id": sid,
               "bindings": deepcopy(self.bindings), "strategy": strategy, "max_reads": budget,
               "cost_loss_units": float(cost), "initial": probabilities, "current": probabilities,
               "steps": [], "status": "READY", "stop_reason": None}
        run["decision"] = self._decision(run)
        if run["decision"]["action_id"] is None:
            run.update(status="STOPPED", stop_reason=run["decision"]["reason"])
        self._save(run)
        return self.view(run)

    def action(self, run_id, payload):
        if set(payload) - {"action", "expected_revision"}:
            raise ValueError("Research actions accept action and expected_revision only")
        action = payload.get("action", "next")
        if action not in ("next", "run", "stop"):
            raise ValueError("Unknown research action")
        run = self._load(run_id)
        revision = payload.get("expected_revision")
        if type(revision) is not int or revision != run["revision"]:
            raise ValueError("STALE_REVISION: reload view before issuing another action")
        if run["status"] == "STOPPED":
            return self.view(run)
        if action == "stop":
            run.update(status="STOPPED", stop_reason="user_stop", revision=run["revision"] + 1, updated_at=now())
            self._save(run)
            return self.view(run)
        while run["status"] != "STOPPED":
            # Work on a copy; a failed f call cannot publish a successful after.
            updated = deepcopy(run)
            selected = run["decision"]["action_id"]
            edge = next(e for e in self.graph[run["state_id"]] if e["action_id"] == selected)
            next_row = self._row(edge["next_state_id"])
            before_ids = set(self._row(run["state_id"])["used_input_ids"])
            if set(next_row["used_input_ids"]) - before_ids != {edge["body_input_id"]}:
                raise ValueError("Archived reveal does not add exactly the selected body")
            after = predictor.predict(self.f, next_row).tolist()
            if any(abs(a-b) > 1e-10 for a,b in zip(after, edge["prediction_after"])):
                raise ValueError("Frozen f recomputation differs from archived transition")
            input_record = self.records[edge["body_input_id"]]
            if input_record["channel"] != "body" or input_record["url"] != edge["body_url"]:
                raise ValueError("Selected archive body identity mismatch")
            resolved, resolution = predictor.resolve(next_row)
            _, routed = scoped_reviewed_f.route(resolved, self.f["head"]["mode"])
            step = {"step": len(run["steps"]) + 1, "action_id": selected,
                    "decision": deepcopy(run["decision"]), "state_before": run["state_id"],
                    "state_after": edge["next_state_id"], "prediction_before": run["current"],
                    "prediction_after": after, "f_executed": True,
                    "revealed_input_id": edge["body_input_id"], "used_input_ids": next_row["used_input_ids"],
                    "modeled_evidence_count": sum(r["lane"] != "unsupported" for r in routed),
                    "body": {"input_id": input_record["input_id"], "title": input_record["title"],
                             "url": input_record["url"], "text": input_record["text"],
                             "sha256": sha256(input_record["text"].encode("utf-8")).hexdigest()},
                    "claims": [e["claim"] for e in next_row["evidence_events"] if e["input_id"] == edge["body_input_id"]],
                    "resolution": resolution, "executed_at": now(), "network_requests": 0}
            updated["steps"].append(step)
            updated.update(state_id=edge["next_state_id"], current=after,
                           revision=run["revision"] + 1, updated_at=now())
            updated["decision"] = self._decision(updated)
            if updated["decision"]["action_id"] is None:
                updated.update(status="STOPPED", stop_reason=updated["decision"]["reason"])
            self._save(updated)
            run = updated
            if action == "next":
                break
        return self.view(run)

    def get(self, run_id, report=False):
        run = self._load(run_id)
        if not report:
            return self.view(run)
        result = {"schema_version": "footballmind-research-g-report-v1", "view": self.view(run),
                  "trace": run, "evaluation": None}
        if run["status"] == "STOPPED":
            # Labels enter only terminal evaluation, after all policy decisions are frozen.
            case = next(c for c in read(self.files["cases"]) if c["fixture"]["physical_match_id"] == run["match_id"])
            outcome = case["outcome"]
            before = -math.log(run["initial"][outcome]); after = -math.log(run["current"][outcome])
            reduction = before-after
            result["evaluation"] = {"scope": "historical_reconstruction_diagnostic", "outcome": outcome,
                                    "loss_before": before, "loss_after": after,
                                    "loss_reduction": reduction,
                                    "net_realized_value": reduction-len(run["steps"])*run["cost_loss_units"],
                                    "prospective_training_eligible": False}
        return result

    def view(self, run):
        fixture = self.fixtures[run["match_id"]]
        row = self._row(run["state_id"])
        def prediction(p):
            return dict(zip(("home_win_probability", "draw_probability", "away_win_probability"), p),
                        model_version=self.f["schema"], prediction_cutoff=row["decision_at"],
                        information_mode="historical_reconstruction")
        after = run["current"] if run["steps"] else None
        change = "not_applicable" if after is None else (
            "changed" if any(abs(a-b) > 1e-9 for a,b in zip(after, run["initial"])) else "unchanged_after_f")
        return {"schema_version": SCHEMA, "run_id": run["run_id"], "revision": run["revision"],
                "match_id": run["match_id"], "mode": "historical_research", "is_live": False,
                "time_mode": "archive_reconstruction", "information_mode": "historical_reconstruction",
                "source_label": "历史研究运行：已训练 g 自主选择，归档正文逐篇披露，冻结 f 重算",
                "external_requests_enabled": False, "production_ready": False,
                "prediction_cutoff": row["decision_at"],
                "match": {"home": fixture["home_name"], "away": fixture["away_name"],
                          "kickoff": fixture["kickoff"], "competition": fixture["competition"]},
                "lifecycle": {"state": run["status"], "reason": run["stop_reason"],
                              "is_terminal": run["status"] == "STOPPED"},
                "prediction": {"before": prediction(run["initial"]), "after": prediction(after) if after else None,
                               "current": prediction(run["current"]), "change_status": change,
                               "delta": {k: b-a for k,a,b in zip(("home_win_probability", "draw_probability", "away_win_probability"), run["initial"], after)} if after else None},
                "f": {"status": "updated" if after else "initial_prediction_only", "model_version": self.f["schema"],
                      "artifact_sha256": self.bindings["f"], "executed_this_run": True,
                      "consumed_body_input_ids": [s["revealed_input_id"] for s in run["steps"]],
                      "used_input_ids": row["used_input_ids"],
                      "modeled_evidence_count": run["steps"][-1]["modeled_evidence_count"] if after else None},
                "g": {"status": "research_ready", "model_version": self.g["schema"],
                      "artifact_sha256": self.bindings["g"], "calibration_status": "not_established",
                      "selection_authority": "fixed_order" if run["strategy"] == "fixed_order" else "research_g",
                      "online_selection_allowed": False, "research_selection_allowed": True,
                      "strategy": run["strategy"], "candidate_availability": "returned",
                      "candidate_actions": deepcopy(run["decision"]["candidates"]),
                      "proposed_action_id": run["decision"]["action_id"] if run["status"] != "STOPPED" else None,
                      "decision_reason": run["decision"]["reason"], "stop_reason": run["stop_reason"]},
                "budget": {"max_reads": run["max_reads"], "used_reads": len(run["steps"]),
                           "remaining_reads": run["max_reads"]-len(run["steps"]),
                           "cost_per_read": run["cost_loss_units"], "spent": len(run["steps"])*run["cost_loss_units"],
                           "unit": "natural_log_loss_equivalent_not_dollars"},
                "steps": deepcopy(run["steps"]), "model_bindings": {k: self.bindings[k] for k in ("f", "g")},
                "limitations": ["历史重建与既有开发留出集，不是新比赛在线效果验证",
                                "正文语义复用归档审阅；本次未调用语义 LLM",
                                "研究 g 未校准，停止代表估计净值非正或预算/候选用尽，不代表信息客观无价值"]}
