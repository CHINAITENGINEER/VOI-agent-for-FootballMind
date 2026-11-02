"""Frozen terminal commitments bind verified history, not later snapshots."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.ambiguity_value import FiniteAmbiguityDecisionModel
from football_agent.audit import digest
from football_agent.decision_loss import FiniteDecisionLossModel
from football_agent.decision_risk_experiment import build_selector, proposal, ConstructedTools, task
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_ambiguity import AmbiguityRiskSelector
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_verify import verify


NOW = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)


def execute(root, outcome="no"):
    base = build_selector(k_weight=1)
    models = {}
    for label, truth in (("assume_yes", 1), ("assume_no", 0)):
        raw = base.model.manifest()
        raw["prior"] = {theta: float(theta == f"K{truth}J1S1") for theta in raw["prior"]}
        models[label] = FiniteDecisionLossModel.from_manifest(raw)
    selector = AmbiguityRiskSelector(FiniteAmbiguityDecisionModel(models), base.bindings,
                                    task_id=base.task_id, loss_budget=.2)

    class Provider:
        def __init__(self): self.calls = 0
        def configuration(self): return {"provider": "constructed_terminal_binding", "configured": True}
        def decide(self, public):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("Frozen policy must finish independently of a second LLM call")
            return {"response": proposal(["fill_j", "verify_k"]), "receipt": {"real_provider_call": False}}

    provider = Provider()
    tools = ConstructedTools(root, {"verify_k": outcome}, clock=lambda: NOW)
    workflow = InquiryWorkflow(root/"runs", provider, tools, clock=lambda: NOW,
                               protocol=EvidenceStateProtocol(), selector=selector)
    started = workflow.start(task(NOW, selector.binding), max_requests=3, max_llm_calls=1)
    run = workflow.run(started["run_id"])
    return workflow, selector, provider, workflow.get(run["run_id"])


def rehash_and_save(workflow, run):
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({key: value for key, value in event.items() if key != "event_hash"})
        previous = event["event_hash"]
    # Keep final summary consistent so the attack tests semantic replay rather
    # than merely a stale hash or a final-snapshot mismatch.
    for key, value in run["events"][-1]["snapshot"].items():
        run[key] = deepcopy(value)
    workflow.store.save(run)
    assert workflow.store.verify(run["run_id"])["status"] == "OK"


def replace_terminal(run, terminal):
    reached = False
    for event in run["events"]:
        if event["type"] == "inquiry_frozen_terminal":
            event["data"] = deepcopy(terminal)
            reached = True
        if reached:
            event["snapshot"]["decisions"][-1] = deepcopy(terminal)


@pytest.mark.parametrize("outcome,state,terminal", [("no", "STOPPED", "K_no__J_yes"), ("failure", "DEFERRED", None)])
def test_terminal_replay_counts_deterministic_execution_separately_from_llm(tmp_path, outcome, state, terminal):
    _, _, provider, run = execute(tmp_path, outcome)
    assert provider.calls == 1 and run["budget"]["llm_calls"] == 1
    assert run["state"] == state
    frozen = next(e for e in run["events"] if e["type"] == "inquiry_frozen_terminal")
    decision = frozen["data"]
    assert decision["decision_origin"] == "deterministic_frozen_policy"
    assert decision["llm_proposal"] is None
    if terminal:
        assert decision["terminal_decision"]["decision"] == terminal
        assert decision["terminal_decision"]["policy_followed"]
    else:
        assert decision["terminal_decision"] is None
        assert decision["risk_assessment"]["state"]["status"] == "MODEL_SET_MISSPECIFIED"
    proof = verify(tmp_path, run["run_id"])
    assert proof["llm_decisions_verified"] == 1
    assert proof["deterministic_terminal_decisions_verified"] == 1
    assert proof["frozen_terminal_decisions_replayed"] == 1
    assert proof["decisions_verified"] == proof["selector_decisions_replayed"] == 2
    assert proof["real_completed_llm_turns"] == 0 and proof["new_network_requests"] == 0


def test_rehashed_later_snapshot_cannot_replace_an_already_verified_commitment(tmp_path):
    workflow, selector, _, run = execute(tmp_path)
    terminal_event = next(e for e in run["events"] if e["type"] == "inquiry_frozen_terminal")
    first = next(e for e in run["events"] if e["type"] == "inquiry_decision")
    assert first["data"]["terminal_policy_commitment"]["response_policy"]["no"] == "K_no__J_yes"
    reached = False
    for event in run["events"]:
        reached |= event["type"] == "inquiry_frozen_terminal"
        if reached:
            event["snapshot"]["decisions"][0]["terminal_policy_commitment"]["response_policy"]["no"] = "K_yes__J_yes"
    # Form a self-consistent *wrong* terminal against the modified snapshot.
    # The unchanged earlier event is the authoritative query-time commitment.
    selection_state = deepcopy(terminal_event["snapshot"])
    selection_state["decisions"] = selection_state["decisions"][:-1]
    changed = selector.settle_after_return(selection_state)
    assert changed["terminal_decision"]["decision"] == "K_yes__J_yes"
    changed.update(at=terminal_event["data"]["at"], candidates=[], claim_updates=[],
                   knowledge_changed=False, decision_origin="deterministic_frozen_policy")
    replace_terminal(run, changed)
    rehash_and_save(workflow, run)
    with pytest.raises(ValueError, match="previously verified decision prefix"):
        verify(tmp_path, run["run_id"])


def test_nondecision_event_cannot_temporarily_rewrite_verified_history(tmp_path):
    workflow, _, _, run = execute(tmp_path)
    acquired = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_returned")
    acquired["snapshot"]["decisions"][0]["terminal_policy_commitment"]["response_policy"]["no"] = "K_yes__J_yes"
    rehash_and_save(workflow, run)
    with pytest.raises(ValueError, match="previously verified decision prefix"):
        verify(tmp_path, run["run_id"])


def test_rehashed_terminal_cannot_claim_a_different_decision_from_frozen_return(tmp_path):
    workflow, _, _, run = execute(tmp_path)
    terminal = deepcopy(next(e["data"] for e in run["events"] if e["type"] == "inquiry_frozen_terminal"))
    terminal["terminal_decision"]["decision"] = "K_yes__J_yes"
    replace_terminal(run, terminal)
    rehash_and_save(workflow, run)
    with pytest.raises(ValueError, match="verified acquisition commitment"):
        verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("time_case", ["before_return", "after_deadline"])
def test_rehashed_terminal_must_execute_after_return_and_before_deadline(tmp_path, time_case):
    workflow, _, _, run = execute(tmp_path)
    event = next(e for e in run["events"] if e["type"] == "inquiry_frozen_terminal")
    terminal = deepcopy(event["data"])
    stamp = NOW-timedelta(seconds=1) if time_case == "before_return" else datetime.fromisoformat(run["task"]["deadline"])+timedelta(seconds=1)
    terminal["at"] = stamp.isoformat()
    event["at"] = stamp.isoformat()
    replace_terminal(run, terminal)
    rehash_and_save(workflow, run)
    with pytest.raises(ValueError, match="execution time"):
        verify(tmp_path, run["run_id"])
