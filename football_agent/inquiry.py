"""Durable open-information inquiry on the existing RunStore/RawArchive primitives.

The predictor-facing acquisition executor remains unchanged. This coordinator
admits parameterized questions and text evidence, for which no fitted gain model
exists yet. Its selection basis is explicitly bounded exploration, not numerical
VoI. An optional scoped selector admits explicitly declared finite decision
models without relabeling unmodeled text. Source claims remain unverified.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .audit import RunStore, digest, redact
from .inquiry_contract import validate_response, utc
from .inquiry_tools import public_url


class InquiryWorkflow:
    version = "operational-inquiry-v1"

    def __init__(self, root, provider, tools, *, clock=None, protocol=None, selector=None, continuation=None,
                 source_catalog=None, temporal_gate=None):
        self.store, self.provider, self.tools = RunStore(root), provider, tools
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.protocol = protocol
        self.selector = selector
        self.continuation = continuation
        self.temporal_gate = temporal_gate
        self.source_catalog = None
        if getattr(protocol, "requires_continuation", False) and continuation is None:
            raise ValueError("Information choice requires explicit continuation count bounds")
        if source_catalog is not None:
            from .inquiry_choice import validate_catalog
            if not getattr(protocol, "supports_source_catalog", False):
                raise ValueError("Source catalog requires the information-choice protocol")
            self.source_catalog = validate_catalog(source_catalog, tools.allowed_hosts)
        if continuation and selector:
            raise ValueError("LLM continuation admission is separate from deterministic scoped selectors")
        if selector and getattr(protocol, "version", None) != "evidence-state-inquiry-v1":
            raise ValueError("Scoped decision selection requires evidence-state-inquiry-v1")

    def configuration(self):
        config = {"version": self.version, "provider": self.provider.configuration(),
                "tools": self.tools.configuration(), "value_model": {"status": "UNSUPPORTED_OPEN_TEXT",
                "expected_gain": None, "policy": "bounded_exploration"}}
        if self.protocol:
            config["inquiry_protocol"] = self.protocol.version
            if hasattr(self.protocol, "decision_problem"):
                config["decision_problem"] = deepcopy(self.protocol.decision_problem)
        if self.continuation:
            config["continuation_policy"] = self.continuation.configuration()
        if self.source_catalog is not None:
            config["source_catalog"] = deepcopy(self.source_catalog)
        if self.selector:
            config["decision_selector"] = self.selector.configuration()
            if getattr(self.selector, "exploration", None):
                config["inquiry_exploration"] = self.selector.exploration.configuration()
        if self.temporal_gate:
            config["temporal_gate"] = self.temporal_gate.configuration()
        return config

    def _event(self, run, kind, data):
        stamp = utc(self.clock())
        if run["events"] and stamp < utc(run["events"][-1]["at"]):
            raise ValueError("Inquiry clock moved backwards")
        snapshot = deepcopy({key: value for key, value in run.items() if key != "events"})
        event = redact({"seq": len(run["events"]) + 1, "at": stamp.isoformat(), "type": kind, "data": data,
                        "snapshot": snapshot, "previous_hash": run["events"][-1]["event_hash"] if run["events"] else None})
        event["event_hash"] = digest(event)
        run["events"].append(event)
        self.store.save(run)

    def start(self, task, *, max_requests=4, max_llm_calls=5):
        if type(max_requests) is not int or not 1 <= max_requests <= 6:
            raise ValueError("Request limit must be 1..6")
        if type(max_llm_calls) is not int or not 1 <= max_llm_calls <= 8:
            raise ValueError("LLM limit must be 1..8")
        required = {"task_id", "objective", "deadline", "initial_facts", "seed_urls"}
        if set(task) != required or not task["task_id"] or not task["objective"]:
            raise ValueError("Explicit task, facts, seeds and deadline required")
        if not isinstance(task["initial_facts"], list) or not isinstance(task["seed_urls"], list):
            raise ValueError("Task facts/seeds must be lists")
        if len(json.dumps(task, ensure_ascii=False)) > 16000:
            raise ValueError("Initial task is too large")
        if utc(self.clock()) >= utc(task["deadline"]):
            raise ValueError("Operational deadline has passed; no retrospective relabeling")
        task = deepcopy(task)
        problem = getattr(self.protocol, "decision_problem", None)
        if problem and problem["task_id"] != task["task_id"]:
            raise ValueError("Terminal decision problem belongs to another task")
        task["seed_urls"] = [public_url(url, self.tools.allowed_hosts) for url in task["seed_urls"]]
        if self.source_catalog is not None and {item["url"] for item in self.source_catalog} - set(task["seed_urls"]):
            raise ValueError("Configured source entries must be declared task seed URLs")
        run = {"schema_version": self.version, "run_id": str(uuid4()), "task": task,
               "configuration": self.configuration(), "state": "READY", "reason": "STARTED",
               "created_at": utc(self.clock()).isoformat(), "questions": [], "claims": [], "documents": [],
               "receipts": [], "decisions": [], "pending": None, "revisit_at": None,
               "budget": {"max_requests": max_requests, "max_llm_calls": max_llm_calls,
                          "http_attempts": 0, "reserved_requests": 0, "llm_calls": 0,
                          "llm_usage": [], "provider_money": None, "money_status": "not_measured"},
               "prediction": None, "prediction_status": "OPEN_TEXT_UPDATE_NOT_CONFIGURED",
               "evidence_scope": "operational_inquiry_execution_not_effectiveness_evidence", "events": []}
        if self.protocol:
            run.update(self.protocol.public(run))
        with self.store.lock(run["run_id"]):
            self._event(run, "inquiry_started", {"selection_basis": "bounded_exploration",
                        "expected_gain": None, "deadline": task["deadline"]})
        return deepcopy(run)

    def public(self, run):
        public = deepcopy({"task": run["task"], "now": utc(self.clock()).isoformat(),
                         "questions": run["questions"], "claims": run["claims"], "documents": run["documents"],
                         "action_history": [{k: r[k] for k in ("tool", "parameters", "document_id", "status")}
                                            for r in run["receipts"]],
                         "last_decision": run["decisions"][-1] if run["decisions"] else None,
                         "budget": {"remaining_requests": run["budget"]["max_requests"] - run["budget"]["http_attempts"],
                                    "remaining_llm_calls": run["budget"]["max_llm_calls"] - run["budget"]["llm_calls"]},
                         "tools": self.tools.configuration(), "value_model_status": "UNSUPPORTED_OPEN_TEXT",
                         "prediction_status": run["prediction_status"]})
        if self.protocol:
            public.update(self.protocol.public(run))
        if self.selector:
            public["scoped_decision_value"] = self.selector.public_summary(run)
        if self.continuation:
            public["continuation"] = self.continuation.public(run)
        if self.temporal_gate:
            public["temporal_gate"] = self.temporal_gate.public(now=self.clock())
        return public

    def _finish(self, run, state, reason):
        run["state"], run["reason"] = state, reason
        self._event(run, "inquiry_paused_or_stopped", {"reason": reason, "state": state,
                    "value_stop_proven": False})
        return deepcopy(run)

    def get(self, run_id):
        with self.store.lock(run_id):
            return deepcopy(self.store.load(run_id))

    def resume_wait(self, run_id):
        """Explicitly reactivate a due wait; never poll, sleep or refill budgets."""
        if not self.continuation:
            raise ValueError("Explicit wait resume requires a configured continuation policy")
        with self.store.lock(run_id):
            run = self.store.load(run_id)
            if run["state"] != "WAITING":
                return deepcopy(run)
            if run["pending"]:
                return self._finish(run, "DEFERRED", "INTERRUPTED_CALL_REQUIRES_RECONCILIATION")
            if digest(self.configuration()) != digest(run["configuration"]):
                return self._finish(run, "DEFERRED", "CONFIGURATION_CHANGED")
            now = utc(self.clock())
            if now < utc(run["events"][-1]["at"]):
                raise ValueError("Inquiry clock moved backwards")
            if now >= utc(run["task"]["deadline"]):
                return self._finish(run, "STOPPED", "TASK_DEADLINE_REACHED")
            scheduled = run["revisit_at"]
            if now < utc(scheduled):
                return deepcopy(run)
            if run["budget"]["llm_calls"] >= run["budget"]["max_llm_calls"]:
                return self._finish(run, "DEFERRED", "LLM_BUDGET_EXHAUSTED")
            run.update(state="READY", reason="WAIT_WINDOW_REACHED", revisit_at=None)
            self._event(run, "inquiry_wait_resumed", {"scheduled_at": scheduled, "resumed_at": now.isoformat(),
                "budget_refilled": False, "new_source_observations": 0, "publication_confirmed": False})
            return deepcopy(run)

    def inherit_evidence(self, run_id, *, parent_root, parent_run_id, parent_head_hash, document_ids):
        from .inquiry_memory import load_inherited
        if not getattr(self.protocol, "supports_inheritance", False):
            raise ValueError("This inquiry protocol does not support inherited evidence")
        with self.store.lock(run_id):
            run = self.store.load(run_id)
            if run["state"] != "READY" or run["pending"] or run["budget"]["llm_calls"] or run["documents"]:
                raise ValueError("Import evidence once before the first reasoning call")
            now = self.clock()
            if utc(now) >= utc(run["task"]["deadline"]):
                raise ValueError("Evidence import is past the task deadline")
            record = load_inherited(parent_root, parent_run_id, parent_head_hash, document_ids,
                                    now=now, deadline=run["task"]["deadline"], allowed_hosts=self.tools.allowed_hosts,
                                    ancestors=((str(self.store.root.parent), run_id),))
            ref = self.store.put_blob(run_id, record)
            run["documents"] = deepcopy(record["documents"])
            run["evidence_imports"] = [{k: v for k, v in record.items() if k not in {"documents", "parent_verification"}}]
            self._event(run, "inquiry_evidence_inherited", {"input_ref": ref, "document_ids": record["document_ids"],
                        "new_http_attempts": 0, "fresh_observation": False})
            return deepcopy(run)

    def step(self, run_id):
        with self.store.lock(run_id):
            run = self.store.load(run_id)
            if run["state"] in {"STOPPED", "DEFERRED", "WAITING"}:
                return deepcopy(run)
            if run["pending"]:
                # No unknown-outcome request is silently repeated on restart.
                return self._finish(run, "DEFERRED", "INTERRUPTED_CALL_REQUIRES_RECONCILIATION")
            if digest(self.configuration()) != digest(run["configuration"]):
                return self._finish(run, "DEFERRED", "CONFIGURATION_CHANGED")
            if utc(self.clock()) >= utc(run["task"]["deadline"]):
                return self._finish(run, "STOPPED", "TASK_DEADLINE_REACHED")
            if self._settle_after_return(run):
                return deepcopy(run)
            if run["budget"]["llm_calls"] >= run["budget"]["max_llm_calls"]:
                return self._finish(run, "DEFERRED", "LLM_BUDGET_EXHAUSTED")
            public = self.public(run)
            input_ref = self.store.put_blob(run_id, public)
            run["budget"]["llm_calls"] += 1
            run["pending"] = {"kind": "llm", "call_id": str(uuid4()), "input_ref": input_ref}
            self._event(run, "inquiry_reasoning_reserved", {"input_ref": input_ref, "call_id": run["pending"]["call_id"]})
            try:
                reply = self.provider.decide(public)
            except Exception as exc:
                # Keep the reservation/pending marker: billing or response may be unknown.
                self._event(run, "inquiry_reasoning_failed", {"error_type": type(exc).__name__,
                            "error": str(exc)[:300], "call_outcome": "unknown"})
                return self._finish(run, "DEFERRED", "PROVIDER_FAILED_OR_RESPONSE_UNKNOWN")
            output_ref = self.store.put_blob(run_id, reply)
            run["pending"] = None
            run["budget"]["llm_usage"].append(reply.get("receipt", {}))
            self._event(run, "inquiry_reasoning_returned", {"input_ref": input_ref, "output_ref": output_ref})
            if utc(self.clock()) >= utc(run["task"]["deadline"]):
                return self._finish(run, "STOPPED", "LATE_REASONING_RETURN")
            try:
                response = (self.protocol.validate if self.protocol else validate_response)(reply["response"], public)
                if self.temporal_gate:
                    self.temporal_gate.validate_response(response, public)
                prepared = {c["candidate_id"]: self.tools.prepare(c, public) for c in response["candidates"]}
            except Exception as exc:
                self._event(run, "inquiry_response_rejected", {"error_type": type(exc).__name__, "reason": str(exc)[:500]})
                return self._finish(run, "DEFERRED", "INVALID_INQUIRY_RESPONSE")
            questions = {q["question_id"]: q for q in run["questions"]}
            questions.update({q["question_id"]: q for q in response["questions"]})
            run["questions"] = list(questions.values())
            claims = {c["claim_id"]: c for c in run["claims"]}
            claims.update({c["claim_id"]: {**c, "updated_at": utc(self.clock()).isoformat(),
                          "semantic_verification": "llm_extraction_quote_checked_meaning_unverified"}
                           for c in response["claim_updates"]})
            run["claims"] = list(claims.values())
            assessment = self.protocol.advance(run, response) if self.protocol else None
            decision = {**response["decision"], "at": utc(self.clock()).isoformat(),
                        "candidates": response["candidates"], "claim_updates": response["claim_updates"],
                        "selection_basis": "llm_bounded_exploration", "value_status": "UNSUPPORTED_OPEN_TEXT",
                        "expected_gain": None, "cost_in_loss_units": None,
                        "knowledge_changed": bool(response["claim_updates"])}
            if self.selector:
                decision.update(self.selector.select(run, response))
            if assessment is not None:
                decision["hypothesis_assessment"] = assessment
            if self.continuation:
                decision.update(self.continuation.apply(run, response, decision, now=decision["at"]))
            run["decisions"].append(decision)
            self._event(run, "inquiry_decision", decision)
            kind = decision["kind"]
            if kind == "reflect" and getattr(self.protocol, "supports_reflection", False):
                run["reason"] = "EVIDENCE_STATE_UPDATED"
                self._event(run, "inquiry_reflection_completed", {"new_http_attempts": 0,
                            "value_status": "UNSUPPORTED_OPEN_TEXT"})
                return deepcopy(run)
            if kind != "query":
                run["revisit_at"] = decision["revisit_at"]
                reason = {"stop": "LLM_QUALITATIVE_STOP", "defer": "LLM_EVIDENCE_INSUFFICIENT",
                          "wait": "LLM_WAIT_FOR_INFORMATION"}[kind]
                if decision["selection_basis"] in {"model_scoped_myopic_decision_risk", "model_set_shared_policy_minimax",
                                                   "model_set_frozen_terminal_policy"}:
                    reason = "SCOPED_MODEL_" + kind.upper()
                elif self.selector and kind != decision["llm_proposal"]["kind"]:
                    reason = "SCOPED_SELECTOR_" + kind.upper()
                if decision["selection_basis"] == "operational_continuation_feasibility":
                    reason = decision["continuation_check"]["blocked_reason"]
                return self._finish(run, {"stop": "STOPPED", "defer": "DEFERRED", "wait": "WAITING"}[kind],
                                    reason)
            candidate = next(c for c in response["candidates"] if c["candidate_id"] == decision["candidate_id"])
            action = prepared[candidate["candidate_id"]]
            execution_key = digest(action)
            from .inquiry_continuation import duplicate_query_blocked
            if duplicate_query_blocked(run, execution_key):
                return self._finish(run, "DEFERRED", "DUPLICATE_QUERY_REQUIRES_NEW_EVIDENCE_OR_WAIT")
            if run["budget"]["http_attempts"] >= run["budget"]["max_requests"]:
                return self._finish(run, "STOPPED", "REQUEST_BUDGET_EXHAUSTED")
            if self.continuation and utc(self.clock()) >= utc(run["task"]["deadline"]):
                return self._finish(run, "STOPPED", "TASK_DEADLINE_REACHED")
            request_id = str(uuid4())
            run["pending"] = {"kind": "http", "request_id": request_id, "execution_key": execution_key,
                              "action": action, "selected_candidate": candidate}
            if self.protocol:
                run["pending"]["inquiry_contract"] = self.protocol.contract(run, candidate)
            run["budget"]["reserved_requests"] += 1
            self._event(run, "inquiry_acquisition_intent", {**run["pending"], "input_ref": self.store.put_blob(run_id, action)})
            if self.continuation and utc(self.clock()) >= utc(run["task"]["deadline"]):
                run["pending"] = None
                run["budget"]["reserved_requests"] -= 1
                self._event(run, "inquiry_acquisition_cancelled", {"request_id": request_id,
                    "reason": "DEADLINE_BEFORE_TOOL_INVOCATION", "http_attempts": 0})
                return self._finish(run, "STOPPED", "TASK_DEADLINE_REACHED")
            try:
                result = self.tools.execute(action, run_id=run_id, request_id=request_id)
            except Exception as exc:
                self._event(run, "inquiry_acquisition_failed", {"error_type": type(exc).__name__, "reservation_retained": True})
                return self._finish(run, "DEFERRED", "ACQUISITION_REQUIRES_RECONCILIATION")
            result_ref = self.store.put_blob(run_id, result)
            if result.get("request_id") != request_id or result.get("http_attempts") != 1:
                return self._finish(run, "DEFERRED", "INVALID_ACQUISITION_RECEIPT")
            self._accept_return(run, result, action, candidate, execution_key, result_ref)
            return deepcopy(run)

    def _accept_return(self, run, result, action, candidate, execution_key, result_ref):
        document = deepcopy(result["document"])
        inquiry_contract = deepcopy(run["pending"].get("inquiry_contract"))
        run["budget"]["http_attempts"] += result["http_attempts"]
        run["budget"]["reserved_requests"] -= 1
        run["pending"] = None
        late = utc(self.clock()) >= utc(run["task"]["deadline"])
        if late:
            document.update(status="LATE_RETURN", text="", links=[])
        for previous in run["documents"]:
            if previous["status"] == "OK" and document["raw_sha256"] == previous["raw_sha256"]:
                document["duplicate_of"] = previous["document_id"]
                break
        run["documents"].append(document)
        receipt = {"tool": action["tool"], "parameters": {k: candidate[k] for k in ("url", "query")},
                   "document_id": document["document_id"], "status": document["status"],
                   "execution_key": execution_key, "request_id": result["request_id"],
                   "result_ref": result_ref, "http_attempts": result["http_attempts"],
                   "latency_seconds": result["latency_seconds"], "money": result["money"]}
        if inquiry_contract is not None:
            receipt["inquiry_contract"] = inquiry_contract
        run["receipts"].append(receipt)
        self._event(run, "inquiry_acquisition_returned", {**receipt, "output_ref": result_ref})
        if late:
            self._finish(run, "STOPPED", "LATE_ACQUISITION_RETURN")
        else:
            self._settle_after_return(run)

    def _settle_after_return(self, run):
        settle = getattr(self.selector, "settle_after_return", None)
        if settle is None:
            return False
        # Recover a persisted terminal event whose final state transition was
        # interrupted. Never repeat the query or request another LLM response.
        last = run["decisions"][-1] if run["decisions"] else None
        if last and last.get("decision_origin") == "deterministic_frozen_policy":
            decision = last
        else:
            settlement = settle(run)
            if settlement is None:
                return False
            if utc(self.clock()) >= utc(run["task"]["deadline"]):
                self._finish(run, "STOPPED", "TASK_DEADLINE_REACHED")
                return True
            if settlement["kind"] not in {"stop", "defer"}:
                raise ValueError("A frozen response policy must settle to stop or defer")
            decision = {**settlement, "at": utc(self.clock()).isoformat(), "candidates": [],
                        "claim_updates": [], "knowledge_changed": False,
                        "decision_origin": "deterministic_frozen_policy"}
            run["decisions"].append(decision)
            self._event(run, "inquiry_frozen_terminal", decision)
        run["revisit_at"] = None
        state = "STOPPED" if decision["kind"] == "stop" else "DEFERRED"
        reason = "SCOPED_FROZEN_POLICY_EXECUTED" if state == "STOPPED" else "FROZEN_POLICY_SUPPORT_WITHHELD"
        self._finish(run, state, reason)
        return True

    def run(self, run_id):
        while True:
            current = self.step(run_id)
            if current["state"] in {"STOPPED", "DEFERRED", "WAITING"}:
                return current

    def export(self, run_id, path):
        run = self.get(run_id)
        audit = self.store.verify(run_id)
        lines = ["# FootballMind 自主信息调查记录", "", run["task"]["objective"], "",
                 f"运行：`{run_id}`；状态：`{run['state']}` / `{run['reason']}`。", "",
                 ("本记录包含声明假设下的有限决策风险选择；开放文本价值仍未知。" if self.selector else
                  "本记录验证获取与重规划过程；开放文本价值模型未配置，查询依据为受限探索。"),
                 "比赛结果数值预测保持未配置，来源表述保留语义未核验状态。", ""]
        for index, decision in enumerate(run["decisions"], 1):
            lines += [f"## 决策 {index}", "", decision["reason"], "",
                      f"动作：`{decision['kind']}`；选择依据：`{decision['selection_basis']}`。", ""]
            if decision.get("continuation_check"):
                check = decision["continuation_check"]
                lines += [f"后续处理预算准入：{check['admitted']}；本次提议之后可用预算：{check['available_after_proposal']}。",
                          f"操作限制：{check['blocked_reason'] or '未触发'}；此检查不估计信息价值或来源发布时间。", ""]
                if decision.get("proposal_before_admission"):
                    lines.append(f"准入前原始提议：`{decision['proposal_before_admission']['kind']}`；新提取证据仍保留。")
            choice = decision.get("hypothesis_assessment", {}).get("information_choice")
            if choice:
                lines += ["候选信息路径比较：", "", choice["review"]["tradeoff"],
                          "来源覆盖限制：" + choice["review"]["missing_routes"],
                          "以下开销为工具路径最低调用数；目录续读、完整交付成本和信息收益仍需判断。", ""]
                routes = {r["candidate_id"]: r for r in choice["candidate_routes"]}
                for item in choice["review"]["alternatives"]:
                    route = routes[item["candidate_id"]]
                    lines += [f"- 路径 `{item['candidate_id']}`：{item['information_advantage']}",
                              f"  - 来源限制：{item['source_limitations']}",
                              f"  - 最低调用数：{route['minimum_counts']}；满足计数下界：{route['within_count_budget']}。",
                              f"  - 本次未优先原因：{item['not_selected_reason'] or '原始提议选择此路径'}；选择时本次返回仍未观察。"]
                lines.append("")
            if "risk_assessment" in decision:
                state = decision["risk_assessment"]["state"]
                lines += [f"声明模型下的预期损失降低：{decision['expected_gain']}；获取代价：{decision['cost_in_loss_units']}。",
                          f"单位：`{state['cost_unit']}`；真实场景校准：否；开放环境全局停止证明：否。", ""]
            if decision.get("robust_risk_reduction") is not None:
                lines += [f"声明模型集合下的最坏风险降低（含获取代价）：{decision['robust_risk_reduction']}。",
                          "适用范围为查询前冻结的一次获取及共同终止策略，不是统计置信区间。", ""]
            if decision.get("terminal_decision"):
                terminal = decision["terminal_decision"]
                lines += [f"终止决策：`{terminal['decision']}`；声明模型下风险：{terminal['worst_expected_loss']}。",
                          f"按冻结返回策略执行：{terminal.get('policy_followed', False)}；独立事实核验：否。", ""]
            if decision.get("exploration_record"):
                record = decision["exploration_record"]
                lines += [f"有界学习探索：合格候选 {len(record['eligible_ids'])} 个，本次候选选择概率 {record['selected_p']}。",
                          "该概率仅条件于本次已生成的合格候选；信息价值、候选生成概率和反事实收益仍未知。", ""]
            for c in decision["candidates"]:
                marker = "已选择" if c["candidate_id"] == decision["candidate_id"] else "候选"
                lines += [f"- {marker} `{c['tool']}`：{c['query'] or c['url']}",
                          f"  - 理由：{c['rationale']}", f"  - 答案可能影响：{c['answer_could_change']}"]
            lines.append("")
        lines += ["## 实际返回", "", "| 工具 | 状态 | 证据 |", "|---|---|---|"]
        for receipt in run["receipts"]:
            lines.append(f"| {receipt['tool']} | {receipt['status']} | {receipt['document_id']} |")
        lines += ["", "## 当前证据与未知", ""]
        for claim in run["claims"]:
            lines += [f"- `{claim['status']}`：{claim['text']}"]
            for citation in claim["citations"]:
                document = next(d for d in run["documents"] if d["document_id"] == citation["document_id"])
                lines.append(f"  - 来源：[{citation['document_id']}]({document['url']})；对应原文锚点保存在运行 JSON 中。")
        if self.protocol:
            lines += ["", "## 假设与查询反馈", "",
                      "以下状态只反映已提取来源的证据类型和适用范围；语义、时效和预测价值仍待验证。", ""]
            states = {s["hypothesis_id"]: s for s in run["hypothesis_states"]}
            for h in run["hypotheses"]:
                state = states[h["hypothesis_id"]]
                lines.append(f"- {h['statement']}：`{state['status']}`；待补证据类型：{', '.join(state['missing_support_types']) or '已覆盖声明类型，仍非事实确认'}。")
            for review in run["query_reviews"]:
                lines.append(f"- 查询 `{review['request_id']}`：`{review['outcome']}`；目标：{', '.join(review['intent']['hypothesis_ids'])}；数值收益未估计。")
            unreviewed = len(run["receipts"]) - len(run["query_reviews"])
            lines.append(f"- 尚待模型处理的返回：{unreviewed}。已保存的逐次查询反馈不作为反事实效用标签。")
        if run.get("evidence_imports"):
            lines += ["", "## 已有证据复用", ""]
            for inherited in run["evidence_imports"]:
                lines.append(f"- 从运行 `{inherited['parent_run_id']}` 继承 {len(inherited['document_ids'])} 份已核验归档正文；原观察时间保留，本轮新增 HTTP 为零，未视为刷新。")
        wakes = [e for e in run["events"] if e["type"] == "inquiry_wait_resumed"]
        if wakes:
            lines += ["", "## 到时恢复", "", "恢复需要显式调用，不自动安排任务、不补充预算，也不视为资料已发布。"]
            for wake in wakes:
                lines.append(f"- 提议复查时刻：{wake['data']['scheduled_at']}；实际恢复：{wake['data']['resumed_at']}；没有新增证据。")
        if run.get("requirements"):
            lines += ["", "## 推理所需输入", "", "以下是已提取输入的可用状态；输入齐全仍需独立推理，并非命题已经成立。", ""]
            for requirement in run["requirements"]:
                lines.append(f"- `{requirement['hypothesis_id']}`：已具备 {', '.join(requirement['present_types']) or '无'}；剩余缺口 {', '.join(requirement['missing_types']) or '无，待推理'}。")
        if run.get("rule_scope_bindings"):
            lines += ["", "## 规则适用关系的来源见证", "",
                      "以下关联保留规则的赛事主体。程序检查身份字段、原文锚点与词面一致性；实体消歧、关系语义、章节归属和时效仍未独立核验。",
                      "关联仅提供资格推理所需的规则输入，不建立资格结论或轮换意图。", ""]
            for binding in run["rule_scope_bindings"]:
                lines += [f"- `{binding['binding_id']}`：`{binding['competition_id']}` / `{binding['season']}` → `{binding['team_id']}` / `{binding['fixture_id']}`。",
                          f"  - 规则陈述 `{binding['claim_id']}`；比赛背景陈述 `{binding['fixture_claim_id']}`；目标 `{binding['hypothesis_id']}`。",
                          "  - 四角色见证：" + "; ".join(f"{role}={binding[role]['document_id']}" for role in
                              ("rule_edition", "team_in_edition", "fixture_in_edition", "team_in_fixture")),
                          "  - Agent 提供的关联依据：" + binding["explanation"]]
        if run.get("return_reviews"):
            lines += ["", "## 查询前预期与实际返回", "",
                      "核对的是事前声明的状态变化和下一步提案，不是收益真值或策略优劣。", ""]
            for review in run["return_reviews"]:
                lines += [f"- 请求 `{review['request_id']}`：实际分支 `{review['outcome']}`；核对结果 `{review['status']}`。",
                          f"  - 缺口预测一致：{review['gap_forecast_match']}；后续提案一致：{review['continuation_match']}。",
                          f"  - 事前计划：`{review['predicted_continuation']['kind']}`；实际提案：`{review['proposed_continuation']['kind']}`。"]
        if run.get("terminal_state"):
            current = run["terminal_state"]
            lines.extend(["", "## 固定终端问题与实际信息影响", "",
                f"- 当前来源立场：{current['action']}；目标：{current['hypothesis_id']}。",
                "- 此处报告来源陈述，不代表实际轮换、真实首发或独立核验后的真值。",
                "- 任务损失为声明偏好；全动作改善上界不是候选信息价值、当前风险或排序分数。"])
            for review in run.get("decision_impact_reviews", []):
                lines.append(f"- 请求 {review['request_id']}：{review['decision_before']['action']} → "
                             f"{review['decision_after']['action']}；变化归属：{review['attribution']}；实际效用未知。")
        if run.get("return_probability_reviews"):
            from .inquiry_probability import summarize_reviews
            summary = summarize_reviews(run["return_probability_reviews"])
            lines += ["", "## 查询返回概率的前瞻核对", "",
                      "概率来自 Agent 查询前的主观判断，未校准。评分对象是传输与后续提取产生的操作标签，不是信息价值或独立真值。",
                      "提取过程可见旧预测，存在自我确认和漏提取偏差；七类均匀参考未按工具可达类别调整，仅为弱参考。",
                      f"有标签的已执行请求：{summary['observed_reviews']}；评分：{summary['scored_forecasts']}；弃权：{summary['abstentions']}。", ""]
            for review in run["return_probability_reviews"]:
                score = review["evaluation"]["forecast_scores"]
                if score is None:
                    result = "事前弃权，未评分；没有用均匀分布替代"
                else:
                    loss = "+∞" if score["log_loss_status"] == "positive_infinity" else str(score["log_loss_nats"])
                    result = f"Brier 求和={score['raw_multiclass_brier']}；log loss（自然对数）={loss}"
                lines.append(f"- 请求 `{review['request_id']}`：实际 `{review['observed_outcome']}`；{result}。")
            lines += ["", "未选候选及尚未取得标签的请求不评分。实际零概率事件保留无穷损失；这些自适应观察不构成校准或策略收益证明。"]
        transport_label = ("构造传输调用（外部 HTTP 为零）" if run["configuration"]["tools"].get("execution_mode") == "constructed_transport_no_network" else "实际 HTTP 尝试")
        lines += ["", f"{transport_label}：{run['budget']['http_attempts']}；模型调用：{run['budget']['llm_calls']}。",
                  f"事件链核验：{audit['status']}。实际货币费用未报告，保留为空。", ""]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return audit
