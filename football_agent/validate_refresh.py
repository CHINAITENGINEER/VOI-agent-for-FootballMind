"""One bounded real platform refresh plus fixed-cutoff and live audit checks."""
from datetime import datetime, timezone
from pathlib import Path
import json

from .adapters import LocalArchiveTools
from .audit import atomic_write
from .collector_refresh import ExistingCollectorRefresh
from .engine import Workflow
from .platform_tools import PlatformTools
from .types import Budget


def main():
    folder = Path("var/parallel/agent/round2/real-refresh")
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "validation.json"
    correction = folder / "correction.json"
    if correction.exists():
        value = json.loads(correction.read_text(encoding="utf-8"))
        domain = folder / "platform-domain-report.json"
        if domain.exists():
            current = json.loads(domain.read_text(encoding="utf-8"))
            value["latest_domain_service_validation"] = {"run_id": current["run_id"],
                "state": current["state"], "reason": current["reason"],
                "refresh_results": current.get("refresh_results", [])}
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if receipt.exists():
        print(receipt.read_text(encoding="utf-8"))
        return  # Reinspection never repeats a network call.
    tools = PlatformTools(model_id="frequency-spain", time_mode="live", service=ExistingCollectorRefresh())
    workflow = Workflow(tools, "var/parallel/agent/round2/runs")
    cutoff = datetime.now(timezone.utc).isoformat()
    run = workflow.start("spftll:3013698", cutoff,
        Budget(max_calls=12, max_queries=1, max_requests=2, max_cost=2, max_tool_seconds=55, max_elapsed_seconds=100))
    for _ in range(4):
        run = workflow.step(run["run_id"])
    archive = LocalArchiveTools(model_id="frequency-england")
    historical = {"match_id": "spftll:3003895", "prediction_cutoff": "2026-09-20T11:25:00Z"}
    before = archive.execute("read_evidence", historical)
    initial = run["prediction"]
    if run["state"] == "PREDICTED":
        run = workflow.step(run["run_id"], action="query", topic="odds")
    after = archive.execute("read_evidence", historical)
    result = {"verification_kind": "real_platform_collector_network", "run_id": run["run_id"],
              "state": run["state"], "reason": run["reason"], "prediction_cutoff": cutoff,
              "parent_prediction_preserved": run["prediction"] == initial,
              "fixed_historical_evidence_unchanged": before == after,
              "requests": run["budget"]["requests"], "refresh_results": run.get("refresh_results", []),
              "audit": workflow.store.verify(run["run_id"])}
    if run.get("child_run_id"):
        child = workflow.get_run(run["child_run_id"])
        for _ in range(4):
            child = workflow.step(child["run_id"])
        if child["state"] == "PREDICTED":
            child = workflow.step(child["run_id"], action="stop")
        result.update(child_run_id=child["run_id"], child_state=child["state"], child_reason=child["reason"],
                      child_prediction=child["prediction"], child_audit=workflow.store.verify(child["run_id"]))
        atomic_write(folder / "child-report.json", child)
    atomic_write(folder / "parent-report.json", run)
    atomic_write(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
