from datetime import datetime, timedelta, timezone
import math

import pytest

from football_research.acquisition.verify_round9 import (
    acquired_from_state, check_estimate, chronological_partition,
    reference_bin, reference_decision, reference_summary,
)


def rows(count, evsi=.04, gain=-.02, *, same_week=False):
    begin = datetime(2024,1,1,tzinfo=timezone.utc)
    return [{"match_id": str(i), "kickoff": (begin+timedelta(days=0 if same_week else 7*(i%4))).isoformat(),
             "model_evsi": evsi, "delta": gain} for i in range(count)]


@pytest.mark.parametrize("value,bucket",[(0.,0),(.019,0),(.02,1),(.05,2),(.1,3),(1.,3)])
def test_prescheduled_right_boundary(value,bucket):
    assert reference_bin(value) == bucket


def test_negative_predictor_update_gain_stays_negative():
    summary = reference_summary(rows(12),.04)
    assert summary["expected_gain"] == pytest.approx(-.02)
    assert summary["lower_gain"] == pytest.approx(-.02)
    assert summary["support"] == summary["parent_support"] == 12


def test_insufficient_local_support_uses_parent_without_disguising_sample_size():
    observations = rows(24,.01,.01)
    observations[0].update(model_evsi=.04,delta=-.20)
    actual = reference_summary(observations,.04)
    assert actual["local_weight"] == 0
    assert actual["fallback"]
    assert actual["support"] == 1 and actual["parent_support"] == 24
    assert actual["expected_gain"] == pytest.approx(.04-.24/24)


def test_se_accounts_for_local_and_parent_overlap_by_week():
    observations = rows(24)
    for i,r in enumerate(observations):
        r["model_evsi"] = .04 if i<12 else .08
        r["delta"] = r["model_evsi"] + (-.03 if i%2 else .05)
    actual = reference_summary(observations,.04)
    # Both populations have mean residual .01; each week holds the same
    # parity, so each weekly influence is +/- .01 after summing both weights.
    assert actual["residual_correction"] == pytest.approx(.01)
    assert actual["standard_error"] == pytest.approx(math.sqrt(4/3*4*.01**2))
    assert actual["local_weight"] == pytest.approx(12/32)


def test_one_time_block_does_not_invent_mean_uncertainty():
    actual=reference_summary(rows(12,same_week=True),.04)
    assert actual["supported"] and actual["standard_error"] is None
    assert actual["lower_gain"] is None


def test_repeat_of_one_match_does_not_increase_support():
    observations=rows(12)
    observations[-1]["match_id"] = observations[0]["match_id"]
    with pytest.raises(ValueError,match="Duplicate"):
        reference_summary(observations,.04)


def test_mean_and_lcb_can_make_different_decisions_and_respect_budget():
    candidate={"supported":True,"expected_gain":.04,"lower_gain":-.01,
               "action":{"action_id":"results_view","cost":{"money":.02,"requests":1}}}
    assert reference_decision([candidate],risk_level="mean",money=.02,requests=1,lambda_money=.5)["action_id"] == "results_view"
    assert reference_decision([candidate],risk_level="conservative",money=.02,requests=1,lambda_money=.5)["decision"] == "stop"
    assert reference_decision([candidate],risk_level="mean",money=.01,requests=1,lambda_money=0)["decision"] == "stop"


def test_source_missing_is_a_purchased_return():
    state={"groups":{g:{"status":"source_missing" if g in ("elo","recent_form") else "not_acquired"}
                     for g in ("elo","recent_form","xg")}}
    assert acquired_from_state(state) == ("results_view",)


@pytest.mark.parametrize("field,value",[("support",99),("parent_support",99),("lower_gain",.8),("standard_error",.9)])
def test_independent_audit_rejects_changed_support_or_uncertainty(field,value):
    wanted=reference_summary(rows(24),.04)
    recorded={**wanted,"model_evsi":.04,"evsi_bin":1}
    check_estimate(recorded,wanted,.04)
    recorded[field]=value
    with pytest.raises(ValueError):
        check_estimate(recorded,wanted,.04)


def test_chronological_split_counts_matches_keeps_kickoff_ties_and_purges_late_labels():
    begin=datetime(2024,1,1,tzinfo=timezone.utc)
    # The last three matches share a kickoff, so the nominal 80% boundary
    # puts all three in the diagnostic tail instead of splitting the tie.
    samples=[]
    for i,day in enumerate((0,0,0,1,2,3,4,7,7,7)):
        kickoff=begin+timedelta(days=day)
        available=kickoff+timedelta(hours=3)
        if i==6:
            available=begin+timedelta(days=6,hours=1)
        samples.append({"match_id":str(i),"kickoff":kickoff.isoformat(),
                        "result_available_at":available.isoformat(),
                        "state":{"prediction_cutoff":(kickoff-timedelta(days=1)).isoformat()}})
    # Repeated transitions do not change the match-level boundary.
    train,tail,boundary=chronological_partition(samples+samples[:4])
    assert {r["match_id"] for r in tail}=={"7","8","9"}
    assert {r["match_id"] for r in train}==set(map(str,range(6)))
    assert boundary==begin+timedelta(days=6)
