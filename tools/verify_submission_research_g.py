"""Materialize complete research-policy comparisons and frontend API examples."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.research_g import ResearchGApplication


def main():
    out = ROOT / "var/work/fm-submission-g-backend-20261019-v1/evidence"
    out.mkdir(parents=True, exist_ok=True)
    app = ResearchGApplication(out / "runs")
    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    save("catalog.json", app.catalog())
    summaries = []
    for mid in app.match_ids:
        for strategy in ("g_two_step", "g_one_step", "fixed_order"):
            view = app.start({"match_id": mid, "strategy": strategy})
            if not summaries:
                save("example-start.json", view)
                view = app.action(view["run_id"], {"action": "next", "expected_revision": view["revision"]})
                save("example-after-step.json", view)
            done = app.action(view["run_id"], {"action": "run", "expected_revision": view["revision"]})
            assert done["lifecycle"]["is_terminal"]
            report = app.get(done["run_id"], report=True)
            if not summaries:
                save("example-completed-report.json", report)
            summaries.append({"match_id": mid, "match": done["match"], "strategy": strategy,
                              "run_id": done["run_id"], "reads": len(done["steps"]),
                              "stop_reason": done["lifecycle"]["reason"],
                              "change_status": done["prediction"]["change_status"],
                              "selected_actions": [s["action_id"] for s in done["steps"]],
                              "f_calls": 1+len(done["steps"]), "evaluation": report["evaluation"]})
    aggregate = {}
    for strategy in ("g_two_step", "g_one_step", "fixed_order"):
        group = [r for r in summaries if r["strategy"] == strategy]
        aggregate[strategy] = {"matches": len(group), "total_reads": sum(r["reads"] for r in group),
                               "mean_loss_reduction": sum(r["evaluation"]["loss_reduction"] for r in group)/len(group),
                               "mean_net_realized_value": sum(r["evaluation"]["net_realized_value"] for r in group)/len(group)}
    reference = json.loads((ROOT / "var/reviews/fm-g-policy-return-continuation-20261014-v1/policy-replays.json").read_text(encoding="utf-8"))
    policy_names = {"g_two_step": "signed_return_h", "g_one_step": "one_step", "fixed_order": "fixed"}
    for item in summaries:
        old = next(r for r in reference if r["physical_match_id"] == item["match_id"]
                   and r["comparison_policy"] == policy_names[item["strategy"]]
                   and r["budget"] == 3 and r["cost_loss_units"] == 0.001)
        assert item["reads"] == old["expected_reads"]
        assert abs(item["evaluation"]["loss_reduction"]-old["expected_gain"]) < 1e-10
    result = {"scope": "historical_research_development_cohort", "is_prospective": False,
              "external_requests": 0, "training_performed": False, "cost_loss_units": 0.001,
              "existing_policy_reference_matches": len(summaries),
              "runs": summaries, "aggregate": aggregate, "model_bindings": app.catalog()["model_bindings"],
              "limitations": app.catalog()["limitations"]}
    save("verification-summary.json", result)
    print(json.dumps({"output": str(out), "runs": len(summaries), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
