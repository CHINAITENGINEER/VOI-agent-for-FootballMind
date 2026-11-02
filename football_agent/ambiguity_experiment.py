"""Reproducible constructed comparison of point and shared-model decisions.

The scripted provider proposes fill_j; the declared common terminal policy
selects verify_k. Supported and all-model-impossible returns then settle without
a second provider turn. No external HTTP, real CLI call or football gain sample
is part of this mechanism experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

from .ambiguity_value import FiniteAmbiguityDecisionModel
from .audit import atomic_write, digest
from .decision_loss import FiniteDecisionLossModel
from .decision_risk_experiment import build_selector, ConstructedTools, proposal, task
from .inquiry import InquiryWorkflow
from .inquiry_ambiguity import AmbiguityRiskSelector
from .inquiry_evidence_state import EvidenceStateProtocol
from .inquiry_verify import verify


VERSION = "constructed-ambiguity-experiment-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPENDENCY_PACKAGES = ("football_agent", "football_platform", "football_spider")
CASE_RETURNS = (("case-supported-no", "no"), ("case-all-model-impossible", "failure"))


def identification_selector():
    """Two declared priors disagree about K; both assume J and source ready."""
    base = build_selector(k_weight=1.0)
    models = {}
    for name, k_truth in (("assume_yes", 1), ("assume_no", 0)):
        manifest = base.model.manifest()
        manifest["prior"] = {theta: float(theta == f"K{k_truth}J1S1") for theta in manifest["prior"]}
        # Replace the base fair-prior description: its return law and loss table
        # are reused, but the two explicit point priors are different assumptions.
        manifest["assumptions"] = [
            "Constructed declared model; no fitted probability or football calibration.",
            f"This model puts all prior mass on K={k_truth}, J=1 and shared source S=1.",
            "The two models have no declared probability weights; one common deterministic policy must serve both.",
            "Incorrect personnel assertion costs one; deferring a person costs 0.2.",
            "Tool costs are declared terminal-loss penalties, not money or measured latency.",
            "A failure return has zero support in both prior models and invalidates the terminal-policy certificate.",
        ]
        models[name] = FiniteDecisionLossModel.from_manifest(manifest)
    return AmbiguityRiskSelector(FiniteAmbiguityDecisionModel(models), base.bindings,
                                 task_id=base.task_id, loss_budget=0.2)


class ScriptedProvider:
    def __init__(self):
        self.calls = 0

    def configuration(self):
        return {"provider": "scripted_ambiguity_mechanism", "version": VERSION,
                "configured": True, "real_provider": False, "max_calls": 1}

    def decide(self, public):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("Frozen terminal settlement must not request a second scripted proposal")
        return {"response": proposal(["fill_j", "verify_k"], chosen="fill_j"),
                "receipt": {"provider": "scripted_ambiguity_mechanism", "real_provider_call": False}}


def _file_record(path, root):
    return {"path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _freeze_sources(root):
    """Freeze every local Python source in all three dependency packages.

    This deliberately includes optional local modules rather than missing a
    lazy import. External distributions and the Python runtime are versioned
    below; their installed source/binary trees are not claimed as snapshots.
    """
    records = []
    for package in LOCAL_DEPENDENCY_PACKAGES:
        for source in sorted((PROJECT_ROOT/package).rglob("*.py")):
            relative = source.relative_to(PROJECT_ROOT)
            if any(part.startswith(".") for part in relative.parts):
                continue
            body = source.read_bytes()
            snapshot = root/"execution-source"/relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.open("xb") as stream:
                stream.write(body)
            records.append({"path": str(source), "snapshot": str(snapshot.relative_to(root)),
                            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
    return records


def _check_sources(root, records):
    for record in records:
        for path in (Path(record["path"]), root/record["snapshot"]):
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Source differs from the pre-execution snapshot: " + str(path))


def _runtime_versions():
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.platform(),
            "external_distributions": {name: metadata.version(name) for name in
                ("requests", "beautifulsoup4", "jsonschema", "pydantic", "filelock")},
            "external_source_snapshot": False}


def _calculations(selector):
    point = {name: model.evaluate(model.initial_state(), remaining_budget=0.2, horizon=1)
             for name, model in selector.model_set.models.items()}
    shared = selector.model_set.evaluate((), remaining_budget=0.2)
    if any(value["action_id"] != "STOP" for value in point.values()) or shared["action_id"] != "verify_k":
        raise AssertionError("Declared comparison no longer exhibits the intended point/shared-policy distinction")
    values = {value["action_id"]: value for value in shared["candidates"]}
    return {"point_model_plans": point, "shared_model_plan": shared,
            "comparison": {"cost_unit": selector.model_set.cost_unit,
                "point_actions": {name: value["action_id"] for name, value in point.items()},
                "shared_stop_loss": shared["current_risk"],
                "shared_action": shared["action_id"],
                "fill_j_cost": values["fill_j"]["cost"],
                "fill_j_worst_total_risk": values["fill_j"]["worst_expected_risk"],
                "verify_k_cost": values["verify_k"]["cost"],
                "verify_k_worst_total_risk": values["verify_k"]["worst_expected_risk"],
                "verify_k_robust_risk_reduction": values["verify_k"]["robust_net_value"],
                "interpretation": "declared_expected_terminal_risk_plus_cost_not_win_probability"}}


def _run_case(folder, case_id, private_returns, public_task, selector_configuration, clock):
    selector = AmbiguityRiskSelector.from_configuration(deepcopy(selector_configuration))
    provider = ScriptedProvider()
    tools = ConstructedTools(folder, deepcopy(private_returns), clock=clock)
    workflow = InquiryWorkflow(folder/"runs", provider, tools, clock=clock,
                               protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.start(deepcopy(public_task), max_requests=3, max_llm_calls=1)
    run = workflow.run(run["run_id"])
    proof = verify(folder, run["run_id"])
    query, terminal = run["decisions"][0], run["decisions"][-1]
    supported = private_returns["verify_k"] == "no"
    if (provider.calls != 1 or run["budget"]["llm_calls"] != 1
            or tools.calls != ["https://example.com/verify_k"]
            or query["llm_proposal"]["candidate_id"] != "fill_j"
            or query["candidate_id"] != "verify_k"
            or terminal.get("decision_origin") != "deterministic_frozen_policy"
            or terminal["llm_proposal"] is not None
            or proof["real_completed_llm_turns"] != 0
            or proof["frozen_terminal_decisions_replayed"] != 1):
        raise AssertionError("Actual execution differs from the declared one-query experiment")
    if supported:
        if (run["state"] != "STOPPED" or terminal["kind"] != "stop"
                or terminal["terminal_decision"]["decision"] != "K_no__J_yes"
                or not terminal["terminal_decision"]["policy_followed"]):
            raise AssertionError("Supported return did not execute its frozen terminal decision")
    elif (run["state"] != "DEFERRED" or terminal["kind"] != "defer"
            or terminal["terminal_decision"] is not None
            or terminal["risk_assessment"]["state"]["status"] != "MODEL_SET_MISSPECIFIED"):
        raise AssertionError("All-model-impossible return was not withheld")
    trajectory = {"case_id": case_id, "run_id": run["run_id"],
        "scripted_proposal": deepcopy(query["llm_proposal"]),
        "selected_query": {"candidate_id": query["candidate_id"], "selection_basis": query["selection_basis"],
            "robust_risk_reduction": query["robust_risk_reduction"],
            "cost_in_loss_units": query["cost_in_loss_units"],
            "frozen_terminal_policy": deepcopy(query["terminal_policy_commitment"])},
        "actual_receipts": deepcopy(run["receipts"]),
        "terminal": {key: deepcopy(terminal[key]) for key in
            ("kind", "decision_origin", "terminal_decision", "risk_assessment", "llm_proposal")},
        "state": run["state"], "reason": run["reason"]}
    result = {"case_id": case_id, "run_id": run["run_id"],
        "head_hash": run["events"][-1]["event_hash"], "state": run["state"], "reason": run["reason"],
        "proposed_candidate": "fill_j", "executed_candidate": query["candidate_id"],
        "return_status": run["documents"][-1]["status"],
        "terminal_decision": deepcopy(terminal["terminal_decision"]),
        "model_set_status_after_return": terminal["risk_assessment"]["state"]["status"],
        "active_model_ids_after_return": terminal["risk_assessment"]["state"]["plan"]["active_model_ids"],
        "scripted_provider_calls": provider.calls, "real_llm_calls": 0,
        "constructed_tool_calls": len(tools.calls), "external_http_requests": 0,
        "new_football_observations": 0, "football_policy_gain_evidence": 0,
        "budget": deepcopy(run["budget"]), "frozen_terminal_decisions_replayed": 1}
    workflow.export(run["run_id"], folder/"report.md")
    atomic_write(folder/"trajectory.json", trajectory)
    atomic_write(folder/"verification.json", proof)
    atomic_write(folder/"result.json", result)
    return result


def run_experiment(root, *, clock=None):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new experiment root; existing artifacts are never overwritten")
    root.mkdir(parents=True, exist_ok=False)
    clock = clock or (lambda: datetime.now(timezone.utc))
    sources = _freeze_sources(root)
    selector = identification_selector()
    public_task = task(clock(), selector.binding)
    public_task["objective"] += " 本实验的候选提供器是固定脚本；不调用真实语言模型，也不把损失风险解释为比赛胜率。"
    tapes = {case_id: {"verify_k": value, "fill_j": "yes"} for case_id, value in CASE_RETURNS}
    frozen_inputs = {"model-set.json": selector.model_set.manifest(),
        "selector.json": selector.configuration(), "public-task.json": public_task,
        "scripted-proposal.json": proposal(["fill_j", "verify_k"], chosen="fill_j"),
        "private-return-tapes.json": tapes, "calculations.json": _calculations(selector)}
    for name, value in frozen_inputs.items():
        atomic_write(root/name, value)
    manifest = {"schema_version": VERSION, "created_at": clock().isoformat(),
        "scope": "constructed_mechanism_not_empirical_policy_gain_or_claim_of_voi_novelty",
        "cases": [case_id for case_id, _ in CASE_RETURNS],
        "same_model_set_for_both_cases": True, "same_public_task_for_both_cases": True,
        "scripted_provider": ScriptedProvider().configuration(), "max_llm_calls_per_case": 1,
        "max_requests_per_case": 3, "transport": "constructed_transport_no_network",
        "private_return_tapes_are_policy_input": False,
        "external_http_requests": 0, "real_llm_calls": 0, "new_football_observations": 0,
        "football_policy_gain_evidence": 0, "runtime": _runtime_versions(),
        "source_snapshot_scope": "all_local_python_sources_in_dependency_packages",
        "local_dependency_packages": list(LOCAL_DEPENDENCY_PACKAGES), "sources": sources,
        "frozen_inputs": [_file_record(root/name, root) for name in frozen_inputs]}
    atomic_write(root/"manifest.json", manifest)
    _check_sources(root, sources)
    results = []
    for case_id, _ in CASE_RETURNS:
        _check_sources(root, sources)
        results.append(_run_case(root/case_id, case_id, tapes[case_id], public_task,
                                 frozen_inputs["selector.json"], clock))
    _check_sources(root, sources)
    for record in manifest["frozen_inputs"]:
        if _file_record(root/record["path"], root) != record:
            raise ValueError("A pre-execution input changed during the experiment")
    summary = {"schema_version": VERSION, "status": "VERIFIED_CONSTRUCTED_SHARED_POLICY_EXECUTION",
        "cases": results, "comparison": frozen_inputs["calculations.json"]["comparison"],
        "scripted_provider_calls": sum(case["scripted_provider_calls"] for case in results),
        "constructed_tool_calls": sum(case["constructed_tool_calls"] for case in results),
        "external_http_requests": 0, "real_llm_calls": 0, "new_football_observations": 0,
        "football_policy_gain_evidence": 0, "frozen_inputs_verified": True,
        "source_files_verified": len(sources), "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "result_paths": [str((root/case["case_id"]/"result.json").relative_to(root)) for case in results]}
    atomic_write(root/"results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run_experiment(args.root)
    print(json.dumps({key: result[key] for key in
        ("status", "comparison", "scripted_provider_calls", "constructed_tool_calls",
         "external_http_requests", "real_llm_calls", "source_files_verified")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
