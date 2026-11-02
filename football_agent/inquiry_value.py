"""Scoped decision-risk selection for explicitly modeled inquiry actions.

Only registered exact observations enter the assumed finite model. Open text
and unmodeled queries remain unsupported, never assigned zero information value.
"""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

from .audit import digest
from .decision_loss import FiniteDecisionLossModel
from .decision_value import VALUE_TOLERANCE, _number


VERSION = "scoped-inquiry-decision-risk-v1"


def canonical_url(value):
    p = urlsplit(value)
    if p.scheme != "https" or not p.hostname or p.query or p.fragment or p.username or p.password or p.port not in {None, 443}:
        raise ValueError("Registered observations require exact public HTTPS URLs without query/fragment")
    return urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", "", ""))


class DecisionRiskSelector:
    def __init__(self, model, bindings, *, task_id, loss_budget):
        if not isinstance(model, FiniteDecisionLossModel) or not task_id:
            raise ValueError("Explicit terminal decision model and task identity required")
        self.model = model
        self.task_id, self.loss_budget = task_id, _number(loss_budget, "Loss budget")
        self.bindings = deepcopy(bindings)
        if set(bindings) != set(model.tools):
            raise ValueError("Bindings must cover exactly the model actions")
        urls = set()
        for aid, binding in self.bindings.items():
            binding["url"] = canonical_url(binding["url"])
            if binding["url"] in urls:
                raise ValueError("An observed URL must have exactly one model interpretation")
            urls.add(binding["url"])
            tool = model.tools[aid]
            if set(binding["markers"]) | {binding["failure_outcome"]} != set(tool.outcomes):
                raise ValueError("Explicit markers plus failure must cover all outcomes")
            if binding["failure_outcome"] in binding["markers"] or any(not isinstance(v, str) or not v for v in binding["markers"].values()):
                raise ValueError("Nonempty exact text markers required for successful returns")
            if len(set(binding["markers"].values())) != len(binding["markers"]):
                raise ValueError("Successful outcome markers must be distinct")
            required = {"subject", "scope", "scope_key", "required_evidence", "hypothesis_id", "statement", "hypothesis_type", "question_id"}
            if set(binding["hypothesis"]) != required or binding["evidence_type"] not in binding["hypothesis"]["required_evidence"]:
                raise ValueError("Each model action needs an explicit hypothesis/evidence binding")

    def configuration(self):
        return {"version": VERSION, "model": self.model.manifest(), "bindings": deepcopy(self.bindings),
                "task_id": self.task_id, "loss_budget": self.loss_budget, "horizon": 1,
                "support_scope": "declared_assumed_model_not_calibrated_open_text"}

    @classmethod
    def from_configuration(cls, config):
        if config.get("version") != VERSION or config.get("horizon") != 1:
            raise ValueError("Unsupported inquiry decision selector version or horizon")
        return cls(FiniteDecisionLossModel.from_manifest(config["model"]), config["bindings"],
                   task_id=config["task_id"], loss_budget=config["loss_budget"])

    def extract_observations(self, run):
        """Map acquired evidence without assuming which probability model holds."""
        if run["task"]["task_id"] != self.task_id:
            raise ValueError("Decision model belongs to another task")
        by_url = {b["url"]: aid for aid, b in self.bindings.items()}
        history, observations, spent = [], [], 0.0
        for doc in run["documents"]:
            aid = by_url.get(doc["url"])
            if aid is None:
                raise ValueError("Acquired document has no declared observation model")
            binding = self.bindings[aid]
            if doc["status"] == "OK":
                matches = [token for token, marker in binding["markers"].items() if doc["text"].strip() == marker]
                if len(matches) != 1:
                    raise ValueError("Actual text does not identify a unique registered return")
                token = matches[0]
            elif doc["status"] in {"FAILED", "NO_DATA", "INVALID_CONTENT"}:
                token = binding["failure_outcome"]
            else:
                raise ValueError("Late or unsupported observation cannot update decision risk")
            history.append((aid, token))
            observations.append({"document_id": doc["document_id"], "action_id": aid, "outcome": token,
                                 "document_hash": digest(doc), "mapping": "exact_registered_text_or_failure_status"})
        # Imported observations update belief; only current receipts charge this run.
        for receipt in run["receipts"]:
            doc = next(d for d in run["documents"] if d["document_id"] == receipt["document_id"])
            aid = by_url[doc["url"]]
            spent += self.model.tools[aid].cost
        if spent > self.loss_budget + VALUE_TOLERANCE:
            raise ValueError("Current acquisition charges exceed the declared loss budget")
        return tuple(history), observations, spent

    def state(self, run):
        history, observations, spent = self.extract_observations(run)
        state = self.model.initial_state()
        for action_id, token in history:
            state = self.model.observe(state, action_id, token)
        return state, observations, spent

    def public_summary(self, run):
        result = {"version": VERSION, "model_hash": digest(self.model.manifest()), "cost_unit": self.model.cost_unit,
                  "evidence_class": "SYNTHETIC_ASSUMPTIONS", "model_is_calibrated": False,
                  "global_open_world_stop_proven": False, "horizon": 1,
                  "registered_queries": [{"action_id": aid, **deepcopy(self.bindings[aid])} for aid in sorted(self.bindings)],
                  "assumptions": list(self.model.assumptions)}
        try:
            state, observations, spent = self.state(run)
            result.update(status="SUPPORTED_DECLARED_MODEL_STATE", observations=observations, spent_loss_penalty=spent,
                          remaining_loss_budget=max(0.0, self.loss_budget-spent),
                          terminal_decision=self.model.terminal_decision(state),
                          plan=self.model.evaluate(state, remaining_budget=max(0.0, self.loss_budget-spent), horizon=1))
        except ValueError as exc:
            result.update(status="UNSUPPORTED_MODEL_STATE", reason=str(exc), plan=None, terminal_decision=None)
        return result

    def match_candidate(self, candidate, hypotheses):
        if candidate["tool"] != "read_url":
            return None
        try:
            url = canonical_url(candidate["url"])
        except (TypeError, ValueError):
            return None
        for aid, binding in self.bindings.items():
            if url != binding["url"]:
                continue
            intent, expected = candidate["inquiry_intent"], binding["hypothesis"]
            if (intent["hypothesis_ids"] != [expected["hypothesis_id"]]
                    or intent["expected_evidence"] != [binding["evidence_type"]]
                    or any(intent[key] != expected[key] for key in ("subject", "scope", "scope_key"))
                    or hypotheses.get(expected["hypothesis_id"]) != expected):
                return None
            return aid
        return None

    def select(self, run, response):
        raw = response["decision"]
        summary = self.public_summary(run)
        assessment = {"version": VERSION, "model_hash": summary["model_hash"], "horizon": 1,
                      "state": summary, "candidate_values": [], "global_open_world_stop_proven": False,
                      "stop_scope": "one_step_registered_model_only"}
        fallback = {**deepcopy(raw), "selection_basis": "llm_bounded_exploration", "value_status": "UNSUPPORTED_OPEN_TEXT",
                    "expected_gain": None, "cost_in_loss_units": None, "llm_proposal": deepcopy(raw), "risk_assessment": assessment}
        if summary["status"] != "SUPPORTED_DECLARED_MODEL_STATE" or raw["kind"] == "reflect":
            return fallback
        plan = summary["plan"]
        values = {item["action_id"]: item for item in plan["candidates"]}
        hypotheses = {h["hypothesis_id"]: h for h in run["hypotheses"]}
        supported, unsupported = [], []
        for candidate in response["candidates"]:
            aid = self.match_candidate(candidate, hypotheses)
            if aid is None:
                item = {"candidate_id": candidate["candidate_id"], "status": "UNSUPPORTED", "expected_gain": None}
                unsupported.append(candidate["candidate_id"])
            else:
                value = values[aid]
                item = {"candidate_id": candidate["candidate_id"], "action_id": aid, "status": "MODELED" if value["feasible"] else "UNAVAILABLE",
                        "model_value": value}
                if value["feasible"]:
                    supported.append((candidate, value))
            assessment["candidate_values"].append(item)
        assessment["unsupported_candidate_ids"] = unsupported
        remaining = run["budget"]["max_requests"] - run["budget"]["http_attempts"]
        if remaining <= 0:
            return {**fallback, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "请求额度已用完；保留当前决策风险，不将预算终止解释为信息价值为零。"}
        positive = [(c, v) for c, v in supported if v["immediate_net_value"] > VALUE_TOLERANCE]
        if positive:
            candidate, value = min(positive, key=lambda pair: (pair[1]["expected_total_risk"], pair[0]["candidate_id"]))
            return {**fallback, "kind": "query", "candidate_id": candidate["candidate_id"], "revisit_at": None,
                    "reason": "在已建模且已提出的候选中，选择当前期望决策损失加获取代价最小的查询；未建模查询仍保留未知价值。",
                    "selection_basis": "model_scoped_myopic_decision_risk", "value_status": "DECLARED_ASSUMPTIONS_ONLY",
                    "expected_gain": value["immediate_evsi"], "cost_in_loss_units": value["cost"]}
        if unsupported:
            assessment["stop_certificate"] = "WITHHELD_UNMODELED_CANDIDATES"
            if raw["kind"] == "query" and raw["candidate_id"] in unsupported:
                return fallback
            return {**fallback, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "已建模候选没有正的一步净收益，但仍有未建模候选；暂缓，保留开放查询价值未知。"}
        profitable_unproposed = [v["action_id"] for v in values.values() if v["feasible"] and v["immediate_net_value"] > VALUE_TOLERANCE]
        if profitable_unproposed:
            assessment["stop_certificate"] = "WITHHELD_VALUABLE_REGISTERED_ACTION_NOT_PROPOSED"
            assessment["unproposed_positive_actions"] = profitable_unproposed
            return {**fallback, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "注册模型仍有正的一步净收益动作，但本次没有对应的有效查询提议；保留该缺口。"}
        feasible = [v for v in values.values() if v["feasible"]]
        assessment["stop_certificate"] = "NO_POSITIVE_ONE_STEP_VALUE_IN_REGISTERED_MODEL" if feasible else "NO_FEASIBLE_REGISTERED_ACTION"
        return {**fallback, "kind": "stop", "candidate_id": None, "revisit_at": None,
                "reason": "在声明的有限模型、当前预算及一步视野内，没有值得执行的注册查询；此结论不覆盖未发现的开放信息。",
                "selection_basis": "model_scoped_myopic_decision_risk", "value_status": "DECLARED_ASSUMPTIONS_ONLY"}
