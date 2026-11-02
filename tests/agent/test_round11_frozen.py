from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_agent.audit import RunStore, stored_digest
from football_agent.query_decision_card import build_query_decision_cards
from football_agent.verify_round11_frozen import compare_prediction_updates, select_cases


def constructed_run(*, acquire=True, missing=False, failed=False):
    """Explicitly constructed trace for boundary tests, not empirical evidence."""
    state = {"match_id": "constructed", "prediction_cutoff": "2026-01-01T00:00:00Z", "information_mode": "historical_reconstruction",
             "base_facts": {}, "groups": {}, "mask": {"elo": False, "recent_form": False, "xg": False},
             "evidence_versions": {}, "quality": {"group_statuses": {"elo": "not_acquired", "recent_form": "not_acquired", "xg": "not_acquired"}},
             "action_history": [], "budget": {}, "state_version": "constructed"}
    prediction = {"match_id": "constructed", "model_version": "constructed-joint", "probabilities": {"home": .4, "draw": .3, "away": .3},
                  "observed_return_categories": {}}
    candidate = {"action_id": "xg_view", "group_ids": ["xg"], "status": "AVAILABLE", "source": "constructed",
        "dependency_key": "shared_response", "cost_scope": "simulated_reveal", "cost_vector": {"money": .03, "requests": 1},
        "value": {"status": "OK", "expected_loss_improvement": .04, "lower_gain": -.01, "cost_in_loss_units": .015,
            "net_gain": .025, "conservative_net_gain": -.025, "support": 13, "parent_support": 20,
            "effective_support": 17., "support_status": "supported_local", "gate_reason": "eligible",
            "estimator_version": "constructed-calibrator", "research_estimate": {"model_evsi": .08, "mean_net_gain": .025,
                "lower_net_gain": -.025, "planned_action": "xg_view" if acquire else None, "acquired_action_ids": [],
                "lambda_cost": {"money": .5}, "physical_budget": {"limits": {"money": .05, "requests": 2}, "spent": {"money": 0., "requests": 0}}}}}
    run = {"run_id": "constructed-run", "match_id": "constructed", "prediction_cutoff": state["prediction_cutoff"], "risk_mode": "mean" if acquire else "conservative",
           "state": "PREDICTED", "reason": "PREDICTED", "visible_state": state, "prediction": prediction, "history": [],
           "configuration": {"backend": {"artifact_hashes": {"predictor": "constructed-model-hash", "estimator": "constructed-value-hash"}, "reference_joint": {}}},
           "budget": {"max_simulated_cost": .05, "simulated_cost": 0., "max_actions": 2, "actions": 0}, "events": []}
    def event(kind, data):
        e = {"seq": len(run["events"]) + 1, "type": kind, "at": "2026-01-01T00:00:00Z", "data": deepcopy(data),
             "snapshot": deepcopy({k: v for k, v in run.items() if k != "events"}),
             "previous_hash": run["events"][-1]["event_hash"] if run["events"] else None}
        e["event_hash"] = stored_digest(e)
        run["events"].append(e)
    event("prediction_updated", {"after": prediction})
    event("value_estimated", {"candidates": [candidate]})
    if acquire:
        event("action_selected", {"action_id": "xg_view"})
        updated = deepcopy(state)
        if not missing and not failed:
            updated["mask"]["xg"] = True
            updated["groups"]["xg"] = {"status": "observed", "values": {"constructed_stat": 1.}}
        updated["quality"]["group_statuses"]["xg"] = "source_missing" if missing or failed else "observed"
        receipt = {"status": "PROVIDER_ERROR" if failed else "NO_DATA" if missing else "OK", "action_id": "xg_view", "request_id": "constructed-request",
                   "visible_state": updated, "cost": {"amount": .03, "unit": "simulation_credit", "status": "simulated"}, "evidence_refs": []}
        run["history"].append({"action_id": "xg_view", "result": receipt})
        run["visible_state"] = updated
        run["budget"].update(simulated_cost=.03, actions=1)
        event("acquisition_result", run["history"][0])
        if not failed:
            run["prediction"] = {**prediction, "probabilities": {"home": .5, "draw": .25, "away": .25},
                                 "observed_return_categories": {"xg_view": "missing" if missing else "xg2"}}
            event("prediction_updated", {"after": run["prediction"]})
    run.update(state="DEFERRED" if failed else "STOPPED", reason="JOINT_OBSERVATION_MODEL_MISMATCH" if failed else "BUDGET" if acquire else "VALUE_STOP_NONPOSITIVE_CONSERVATIVE_GAIN")
    event("deferred" if failed else "stopped", {})
    return run


def rehash(run):
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = stored_digest({k: v for k, v in event.items() if k != "event_hash"})
        previous = event["event_hash"]


def test_cards_use_pre_action_snapshot_not_final_paid_evidence():
    run = constructed_run()
    before = deepcopy(run)
    card = build_query_decision_cards(run)
    first, terminal = card["cards"]
    assert first["current"]["probabilities"] == {"home": .4, "draw": .3, "away": .3}
    assert first["current"]["public_mask"]["xg"] is False
    assert terminal["current"]["public_mask"]["xg"] is True
    assert first["receipt"]["prediction_after_return"]["probabilities"] == card["final_probabilities"]
    assert first["candidates"][0]["selection_score"] == .025
    assert "values" not in json.dumps(card)
    assert first["receipt"]["audit"]["seq"] > first["audit"]["seq"]
    assert run == before and card["llm_generated_explanation"] is False


def test_stop_card_has_real_negative_lower_score_and_no_receipt():
    card = build_query_decision_cards(constructed_run(acquire=False))
    decision = card["cards"][0]
    assert decision["decision"]["kind"] == "stop" and decision["receipt"] is None
    assert decision["candidates"][0]["selection_score"] == -.025
    assert decision["candidates"][0]["decision_metric"] == "conservative_net_gain"
    assert card["physical_budget"]["actions"] == 0


@pytest.mark.parametrize("missing,failed,kind", [(True, False, "summary_missing"), (False, True, "provider_failure")])
def test_summary_missing_and_provider_failure_stay_distinct(missing, failed, kind):
    card = build_query_decision_cards(constructed_run(missing=missing, failed=failed))
    receipt = card["cards"][0]["receipt"]
    assert receipt["return_kind"] == kind
    assert receipt["public_mask_after_return"]["xg"] is False
    if missing:
        assert receipt["prediction_after_return"]["observed_return_categories"] == {"xg_view": "missing"}
    else:
        assert receipt["prediction_after_return"] is None
        assert card["cards"][-1]["decision"]["kind"] == "defer"


def test_candidate_tampering_and_run_summary_tampering_fail_audit():
    run = constructed_run()
    run["events"][1]["data"]["candidates"][0]["value"]["support"] += 1
    with pytest.raises(ValueError, match="chain"):
        build_query_decision_cards(run)
    run = constructed_run()
    run["prediction"]["probabilities"] = {"home": 1., "draw": 0., "away": 0.}
    with pytest.raises(ValueError, match="snapshot"):
        build_query_decision_cards(run)


@pytest.mark.parametrize("field", ["hidden_groups", "joint_return_hypothesis"])
def test_rehashed_trace_still_rejects_hidden_or_hypothetical_predecision_state(field):
    run = constructed_run()
    run["events"][1]["snapshot"]["visible_state"][field] = {"private": 1}
    rehash(run)
    with pytest.raises(ValueError):
        build_query_decision_cards(run)


def test_unfinished_and_wrong_receipt_linkage_do_not_make_complete_cards():
    run = constructed_run()
    run["state"] = "WAITING"
    with pytest.raises(ValueError, match="completed"):
        build_query_decision_cards(run)
    run = constructed_run()
    run["events"][3]["data"]["action_id"] = "results_view"
    rehash(run)
    with pytest.raises(ValueError, match="different action"):
        build_query_decision_cards(run)


def schedule():
    folds, rows = [], []
    for n in range(2, 7):
        folds.append({"fold": n, "reference_fold": n-2, "joint_model": {}, "calibrators": {"same_version": {}, "matched_mixed": {}}})
        for strategy in ("base-only", "frozen-version-mean", "frozen-version-lcb", "matched-mixed-mean"):
            for mid, loss in (("z-best", 0.), ("a-first", 99.)):
                rows.append({"fold": n, "strategy": strategy, "match_id": mid, "budget_money": .05, "lambda_money": .5,
                             "kickoff": "2026-01-01T12:00:00Z", "prediction_cutoff": "2025-12-31T12:00:00Z", "log_loss": loss})
    return {"folds": folds}, rows


def test_fixed_fifteen_cases_ignore_gain_result_and_action_filtering():
    report, rows = schedule()
    cases = select_cases(report, rows)
    assert len(cases) == 15 and all(c["first"]["match_id"] == "a-first" for c in cases)
    assert [c["calibration"] for c in cases[:3]] == ["same_version", "same_version", "matched_mixed"]
    assert all(c["reference_fold"] == c["fold"]-2 for c in cases)


@pytest.mark.parametrize("kind", ["lambda", "reference_fold", "missing_policy"])
def test_selection_rejects_tuned_lambda_wrong_frozen_model_or_missing_policy(kind):
    report, rows = schedule()
    if kind == "lambda":
        next(r for r in rows if r["strategy"] == "frozen-version-mean" and r["match_id"] == "a-first")["lambda_money"] = 0.
    elif kind == "reference_fold":
        report["folds"][0]["reference_fold"] = 2
    else:
        rows = [r for r in rows if r["strategy"] != "matched-mixed-mean"]
    with pytest.raises(ValueError):
        select_cases(report, rows)


def test_every_actual_return_update_is_checked_including_missing():
    run = constructed_run(missing=True)
    expected = [{"step": i, "prediction": {**e["snapshot"]["prediction"], "probabilities":
        [e["snapshot"]["prediction"]["probabilities"][k] for k in ("home", "draw", "away")]}}
        for i, e in enumerate(e for e in run["events"] if e["type"] == "prediction_updated")]
    assert compare_prediction_updates(run, expected)["actual_return_prediction_updates_checked"] == 2
    expected[1]["prediction"]["observed_return_categories"] = {}
    with pytest.raises(ValueError, match="encoding"):
        compare_prediction_updates(run, expected)


@pytest.mark.parametrize("strategy", ["calibrated-mean", "calibrated-lcb"])
def test_existing_real_round9_fixed_first_case_cards_match_verified_event_blobs(strategy):
    root = Path(__file__).resolve().parents[2] / "var/parallel/agent/round9/calibrated-joint-bridge-v1"
    if not (root / "acceptance.json").exists():
        pytest.skip("Frozen historical round9 artifact is unavailable in this checkout")
    acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    reference = next(c for c in acceptance["checks"] if c["fold"] == 0 and c["strategy"] == strategy)
    store = RunStore(root / "runs")
    run = store.load(reference["run_id"])
    audit = store.verify(reference["run_id"])
    cards = build_query_decision_cards(run)
    assert cards["audit_head_hash"] == audit["head_hash"]
    assert cards["actual_actions"] == reference["actions"]
    assert cards["final_probabilities"] == run["prediction"]["probabilities"]
    assert cards["cards"][0]["current"]["observed_return_categories"] == {}
