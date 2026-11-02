"""Integration regressions using actual platform and provenance adapters."""
from copy import deepcopy
import hashlib
import json

import pytest

from football_agent.platform_acquisition_context import build_platform_acquisition_context
from football_agent.autonomous_acquisition_step import prepare_acquisition_state, select_acquisition_action

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-136", "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
            "ingested_at": "2026-10-01T08:05:00Z", "home_team_id": "home",
            "away_team_id": "away", "home_score": 3, "away_score": 0,
            "result_available_at": "2026-10-01T20:00:00Z"}


def evidence(sid="lineup", **extra):
    row = {"source_id": sid, "kind": "lineup", "observed_at": "2026-10-01T09:00:00Z",
           "ingested_at": "2026-10-01T09:01:00Z", "freshness": "fresh",
           "availability": "available", "digest": "a" * 64,
           "values": {"rotation_confirmed": False}}
    row.update(extra)
    return row


def candidate(cid="lineup", **extra):
    row = {"candidate_id": cid, "source_ids": [cid], "kind": "football_topic",
           "action_availability": True, "resource_cost": {"http_calls": 1},
           "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "declared_conversion"}}
    row.update(extra)
    return row


def state(rows=None, candidates=None, complete=True, target=None):
    context = build_platform_acquisition_context(target or match(), rows or [])
    return prepare_acquisition_state(context["actor_context"], candidates or [candidate()],
                                     candidate_generation_complete=complete)


def estimate(s, cid="lineup", value=.4, **extra):
    result = {"candidate_id": cid, "state_digest": s["state_digest"],
              "expected_value": value, "unit": "decision_loss", "source": "declared-test-model",
              "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    result.update(extra)
    return result


def select(s, estimates=None, **extra):
    return select_acquisition_action(s, loss_unit="decision_loss", estimates=estimates,
                                     remaining_budget=extra.pop("remaining_budget", {"http_calls": 3}), **extra)


def test_actual_gate_shape_preserves_stale_evidence_and_selects_refresh():
    s = state([evidence(valid_until="2026-10-01T11:00:00Z")])
    out = select(s, {"lineup": estimate(s)})
    assert out["evaluations"][0]["source_states"] == {"lineup": "stale"}
    assert out["evaluations"][0]["evidence_eligibility"] == "defer_unknown"
    assert out["decision"] == "acquire" and out["selected_action_kind"] == "refresh"
    assert out["reservation"]["actual_spend_observed"] is False
    assert out["remaining_budget"] == {"http_calls": 3.0}


def test_missing_target_is_queryable_but_missing_prerequisite_is_deferred():
    s = state()
    assert select(s, {"lineup": estimate(s)})["selected_action_kind"] == "query"
    s = state(candidates=[candidate(requires_source_ids=["fixture-rules"])])
    out = select(s, {"lineup": estimate(s)})
    assert out["decision"] == "defer"
    assert out["evaluations"][0]["blocked_reason"] == "prerequisites_unresolved"


def test_incomplete_menu_allows_supported_positive_action_but_withholds_stop():
    s = state(complete=False)
    assert select(s, {"lineup": estimate(s)})["decision"] == "acquire"
    out = select(s, {"lineup": estimate(s, value=0)})
    assert out["decision"] == "defer"
    assert out["query_value_decision"]["stop_certificate"]["kind"] == "WITHHELD_OPEN_WORLD_VALUE_UNKNOWN"


def test_unknown_cost_competitor_withholds_model_stop():
    s = state(candidates=[candidate(), candidate("weather", utility_cost=None)])
    out = select(s, {"lineup": estimate(s, value=0), "weather": estimate(s, "weather", value=.5)})
    assert out["decision"] == "defer"
    weather = out["evaluations"][1]
    assert weather["assessment"]["expected_value"] is None
    assert weather["assessment"]["reason"] == "utility_cost_unknown"


def test_unknown_action_availability_is_not_default_executable():
    c = candidate()
    c.pop("action_availability")
    s = state(candidates=[c])
    out = select(s, {"lineup": estimate(s)})
    assert out["decision"] == "defer"
    assert out["evaluations"][0]["action_eligibility"] == "unknown"


def test_fresh_evidence_is_usable_and_query_is_not_repeated():
    s = state([evidence()])
    assert s["provenance_gate"]["candidates"][0]["eligibility"] == "eligible"
    out = select(s, {"lineup": estimate(s)})
    assert out["decision"] == "stop"
    assert out["selected_candidate_id"] is None
    assert out["evaluations"][0]["blocked_reason"] == "evidence_already_usable"
    assert out["query_value_decision"]["stop_certificate"]["kind"] == "NO_FEASIBLE_REGISTERED_ACTION"


def test_explicit_refresh_can_still_use_positive_value_on_fresh_evidence():
    s = state([evidence()], candidates=[candidate(action_kind="refresh")])
    out = select(s, {"lineup": estimate(s)})
    assert out["decision"] == "acquire"
    assert out["selected_action_kind"] == "refresh"


def test_result_and_late_rows_do_not_change_state_or_decision():
    original = state([evidence()])
    changed_match = match()
    changed_match.update(home_score=0, away_score=9, label_status="INDEPENDENT_EVALUATOR_GOLD")
    late = evidence("future-news", observed_at="2026-10-02T09:00:00Z", ingested_at="2026-10-02T09:01:00Z")
    changed = state([late, evidence()], target=changed_match)
    assert original == changed
    assert select(original, {"lineup": estimate(original)}) == select(changed, {"lineup": estimate(original)})


def test_visible_update_invalidates_old_estimate_even_if_same_candidate():
    before = state([evidence()])
    after = state([evidence(digest="b" * 64, values={"rotation_confirmed": True})])
    out = select(after, {"lineup": estimate(before)})
    assert out["decision"] == "defer"
    assert out["evaluations"][0]["assessment"]["reason"] == "estimate_state_or_candidate_mismatch"


@pytest.mark.parametrize("field", ["training_cutoff", "available_at"])
def test_future_estimate_not_used_or_projected(field):
    s = state()
    out = select(s, {"lineup": estimate(s, **{field: "2026-10-02T00:00:00Z"})})
    assert out["decision"] == "defer"
    assert out["evaluations"][0]["estimate_support"] == {}
    assert out["evaluations"][0]["assessment"]["expected_value"] is None


def test_stop_certificate_reuses_existing_contract_with_bounded_scope():
    s = state()
    out = select(s, {"lineup": estimate(s, value=.05)})
    cert = out["query_value_decision"]["stop_certificate"]
    assert out["decision"] == "stop"
    assert cert["kind"] == "MODEL_SCOPED_STOP"
    assert cert["scope"] == "declared_finite_model" and cert["global_claim"] is False
    assert cert["evaluated_candidate_ids"] == ["lineup"]


def test_resources_are_not_added_to_loss_cost_or_to_each_other():
    s = state(candidates=[candidate(resource_cost={"http_calls": 1, "local_reads": 4})])
    out = select(s, {"lineup": estimate(s)}, remaining_budget={"http_calls": 100, "local_reads": 3})
    assert out["decision"] == "defer"
    assert out["query_value_decision"]["stop_certificate"]["kind"] == "BUDGET_EXHAUSTED"
    assert out["query_value_decision"]["stop_certificate"]["remaining_budget"] == {"http_calls": 100, "local_reads": 3}


def test_incomparable_units_are_rejected():
    s = state()
    with pytest.raises(ValueError, match="unit"):
        select(s, {"lineup": estimate(s, unit="requests")})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -.1])
def test_invalid_estimate_values_rejected(value):
    s = state()
    with pytest.raises(ValueError, match="finite"):
        select(s, {"lineup": estimate(s, value=value)})


def test_candidate_order_does_not_affect_state_and_ties_are_deterministic():
    a, b = candidate("lineup"), candidate("injury")
    first, second = state(candidates=[a, b]), state(candidates=[b, a])
    assert first == second
    out = select(first, {cid: estimate(first, cid) for cid in ["lineup", "injury"]})
    assert out["selected_candidate_id"] == "injury"


def test_tampered_state_rejected_before_selection():
    s = state()
    s["candidates"][0]["utility_cost"]["amount"] = 0
    with pytest.raises(ValueError, match="digest"):
        select(s)


def test_envelope_is_not_accepted_as_actor_context():
    with pytest.raises(ValueError, match="public actor_context"):
        prepare_acquisition_state(build_platform_acquisition_context(match(), []), [candidate()])


def test_no_estimates_defer_and_no_user_question_is_introduced():
    out = select(state())
    assert out["decision"] == "defer" and out["selected_candidate_id"] is None
    assert "ask_user" not in json.dumps(out)
    assert out["evaluations"][0]["assessment"]["expected_value"] is None


def test_public_catalog_excludes_payload_and_unbound_human_value_fields():
    s = state(candidates=[candidate(description={"result": "future-secret"}, expected_value=999,
                                    question="human chooses injury", evaluator_sidecar={"label": "secret"})])
    assert "future-secret" not in json.dumps(s)
    assert "human chooses" not in json.dumps(s)
    assert select(s)["decision"] == "defer"
