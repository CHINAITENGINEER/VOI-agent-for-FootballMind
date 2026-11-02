"""Constructed event clocks; no HTTP, live outcomes, or learned-g claims."""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from football_research.data_adapters.prospective_outcomes import derive_g_candidates
from football_research.data_adapters.predictor_context_pairs import verify_predictor_pair
from football_spider.storage.raw_archive import RawArchive


def stamp(second):
    return f"2026-10-09T09:00:{second:02d}+00:00"


@pytest.fixture
def pair(tmp_path):
    root = tmp_path / "stream"
    archive = RawArchive(root)
    def raw(payload, at):
        obs = archive.save("agent_trajectory", "agent://test", {}, json.dumps(payload).encode(),
                           status=200, observed_at=datetime.fromisoformat(at))
        return {"raw_path": obs.payload_path, "raw_sha256": obs.content_hash}
    kickoff = "2026-10-10T14:00:00+00:00"
    deadline = "2026-10-10T13:59:00+00:00"
    pb, pa = {"home": .5, "draw": .3, "away": .2}, {"home": .45, "draw": .4, "away": .15}
    state = {"match_id": "provider-1", "base": {"kickoff": kickoff}}
    public = {"state": state, "documents": [], "decision_at": stamp(0),
              "prediction": {"probabilities": [.5, .3, .2], "model_version": "f:one"}}
    menu_payload = {"schema": "prospective-menu-archive-v1", "run_id": "run-1", "step": 0,
        "observed_at": stamp(0), "public_before": public, "policy_arm": "fixed_order",
        "menu": [{"action_id": "read-1", "selected": True, "selection_probability": 1.,
                  "selection_probability_basis": "conditional_on_observed_menu"}]}
    menu = {"discovery_id": "menu-1", "physical_match_id": "m1", "information_mode": "prospective_capture",
        "observed_at": stamp(0), "policy_version": "v2", "original_agent_menu_proven": True,
        "visible_candidates": [{"candidate_id": "candidate-1", "action_id": "read-1"}],
        **raw(menu_payload, stamp(0))}
    response_raw = raw({"article": "constructed input"}, stamp(6))
    doc_input = {"document_id": "doc-1", "observed_at": stamp(6), "available_at": stamp(8),
                 "raw_sha256": response_raw["raw_sha256"]}
    payloads, contexts = {}, {}
    for name, forecast, handoff, start, finish in (("before", 0, 1, 2, 3), ("after", 10, 11, 12, 13)):
        payload = {"schema": "runtime-predictor-context-v3", "run_id": "run-1",
            "snapshot_kind": name + "_action", "state": state,
            "documents": [] if name == "before" else [doc_input],
            "observed_at": stamp(handoff), "available_at": stamp(handoff),
            "forecast_as_of": stamp(forecast), "prediction_cutoff": stamp(forecast),
            "acquisition_deadline": deadline, "availability_basis": "runtime_state_handoff",
            "availability_status": "known", "parent_attempt_id": None if name == "before" else "run-1:attempt:0",
            "prediction_before": pb, "prediction_after": None if name == "before" else pa,
            "predictor_model_versions": ["f:one"]}
        contexts[name] = {**payload, **raw(payload, stamp(start)), "physical_match_id": "m1",
            "snapshot_id": name, "information_mode": "prospective_capture", "captured_at": stamp(finish),
            "raw_context_schema": payload["schema"], "archive_save_started_at": stamp(start),
            "archive_save_returned_at": stamp(finish), "capture_timing_basis": "measured_local_utc_archive_save_return",
            "context_capture_completion_proven": True, "forecast_eligible": True, "recording_scope": "pre_match_context"}
        payloads[name] = payload
    fixture = {"physical_match_id": "m1", "kickoff_at": kickoff, "registered_at": stamp(0),
        "arm_assigned_at": stamp(0), "policy_arm": "fixed_order", "policy_version": "v2",
        "pre_kickoff_frozen": True, "information_mode": "prospective_capture", **raw({"fixture": "m1"}, stamp(0))}
    attempt = {"attempt_id": "run-1:attempt:0", "physical_match_id": "m1", "discovery_id": "menu-1",
        "action_id": "read-1", "candidate_id": "candidate-1", "selected_by_policy": True,
        "policy_arm": "fixed_order", "policy_version": "v2", "selection_basis": "fixed_order",
        "selection_probability": 1., "selection_probability_basis": "conditional_on_observed_menu",
        "information_mode": "prospective_capture", "decision_at": stamp(0), "requested_at": stamp(4),
        "request_started_at": stamp(5), "response_observed_at": stamp(6), "receipt_available_at": stamp(10),
        "status": "returned", "request_count": 1, "acquisition_cost": .01,
        "returned_document_id": "doc-1", **response_raw}
    document = {**doc_input, **response_raw, "physical_match_id": "m1", "information_mode": "prospective_capture",
        "captured_at": stamp(7), "available_at": stamp(11), "capture_status": "known", "document_admitted": True,
        "capture_timing_basis": "measured_local_utc_archive_save_return", "availability_status": "known",
        "availability_basis": "verified_runtime_state_handoff"}
    result = {"physical_match_id": "m1", "kickoff_at": kickoff, "regulation_home_goals": 1, "regulation_away_goals": 1,
        "result": "draw", "result_status": "final", "score_period": "regulation",
        "result_source_url": "https://example.test/constructed-result", "result_observed_at": "2026-10-10T17:00:00+00:00",
        "information_mode": "post_match_outcome", "independent_outcome_verified": True,
        **raw({"synthetic_result": [1, 1]}, "2026-10-10T17:00:00+00:00")}
    return dict(root=root, before=contexts["before"], after=contexts["after"], menu=menu, attempt=attempt,
                fixture=fixture, document=document, result=result, payloads=payloads, raw=raw)


def audit(p):
    return verify_predictor_pair(**{key: p[key] for key in ("root", "before", "after", "menu", "attempt", "fixture")})


def derive(p, *, contexts=None):
    return derive_g_candidates(fixtures=[p["fixture"]], discoveries=[p["menu"]], attempts=[p["attempt"]],
        contexts=[p["before"], p["after"]] if contexts is None else contexts, results=[p["result"]],
        documents=[p["document"]], archive_root=p["root"])[0]


def test_pair_uses_exact_menu_forecast_despite_later_handoff(pair):
    assert audit(pair)["valid"]
    stale = deepcopy(pair["before"])
    stale.update(forecast_as_of=stamp(0).replace("09:00", "08:00"), observed_at="2026-10-09T08:00:01+00:00",
                 prediction_before={"home": .9, "draw": .05, "away": .05})
    result = derive(pair, contexts=[stale, pair["before"], pair["after"]])
    assert result["label_status"] == "eligible", result["exclusion_reason"] if "exclusion_reason" in result else result
    assert result["prediction_before"] == pair["before"]["prediction_before"]
    assert result["realized_gain"] == pytest.approx(__import__("math").log(.4 / .3))
    assert pair["document"]["available_at"] == stamp(11)  # The handoff was not backdated to forecast=10.


@pytest.mark.parametrize("case", ["before_clock", "after_clock", "late_before_save", "late_after_save", "before_prediction",
    "after_prediction", "model", "run", "parent", "forecast", "menu_hash", "sidecar", "document_sha", "capture_unknown",
    "input_late", "extra_document", "malformed_state", "late_cutoff", "outer_schema", "propensity_forged"])
def test_invalid_pair_never_supplies_gain(pair, case):
    name = "before" if case in {"before_clock", "late_before_save", "before_prediction"} else "after"
    row, payload = pair[name], deepcopy(pair["payloads"][name])
    if case == "before_clock": row["forecast_as_of"] = stamp(1)
    elif case == "after_clock": row["available_at"] = stamp(10)
    elif case == "late_before_save": row.update(captured_at=stamp(5), archive_save_returned_at=stamp(5))
    elif case == "late_after_save": row.update(captured_at="2026-10-10T14:00:00+00:00", archive_save_returned_at="2026-10-10T14:00:00+00:00")
    elif case == "before_prediction": row["prediction_before"] = {"home": .9, "draw": .05, "away": .05}
    elif case == "after_prediction": row["prediction_after"] = {"home": .9, "draw": .05, "away": .05}
    elif case == "model": payload["predictor_model_versions"] = ["f:two"]
    elif case == "run": payload["run_id"] = "other-run"
    elif case == "parent": payload["parent_attempt_id"] = "run-1:attempt:9"
    elif case == "forecast": payload["forecast_as_of"] = payload["prediction_cutoff"] = stamp(9)
    elif case == "menu_hash": pair["menu"]["raw_sha256"] = "0" * 64
    elif case == "sidecar": row["archive_save_started_at"] = stamp(11)
    elif case == "document_sha": pair["document"]["raw_sha256"] = "1" * 64
    elif case == "capture_unknown": row["captured_at"] = None
    elif case == "input_late": payload["documents"][0]["available_at"] = stamp(14)
    elif case == "extra_document": payload["documents"].append({"document_id": "not_this_return"})
    elif case == "malformed_state": payload["state"] = None
    elif case == "late_cutoff": payload["acquisition_deadline"] = stamp(12)
    elif case == "outer_schema": row["raw_context_schema"] = "runtime-predictor-context-v99"
    elif case == "propensity_forged": pair["attempt"]["selection_probability"] = .5
    if payload != pair["payloads"][name]:
        row.update(payload)
        row.update(pair["raw"](payload, row["archive_save_started_at"]))
    result = derive(pair)
    assert result["label_status"] == "ineligible"
    assert result["realized_gain"] is None and result["net_gain"] is None


def test_duplicate_exact_forecast_is_ambiguous_and_unknown_remains_unknown(pair):
    result = derive(pair, contexts=[pair["before"], deepcopy(pair["before"]), pair["after"]])
    assert "before_predictor_context_ambiguous" in result["exclusion_reason"]
    pair["attempt"].update(selection_probability=None, selection_probability_basis=None, acquisition_cost=None)
    raw_menu = json.loads(__import__("gzip").decompress((pair["root"] / pair["menu"]["raw_path"]).read_bytes()))
    raw_menu["menu"][0].update(selection_probability=None, selection_probability_basis=None)
    pair["menu"].update(pair["raw"](raw_menu, stamp(0)))
    result = derive(pair)
    assert result["predictor_pair_audit"]["valid"]
    assert {"propensity_missing", "acquisition_cost_unknown"} <= set(result["exclusion_reason"])
    assert result["realized_gain"] is None
