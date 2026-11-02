from copy import deepcopy

import pytest

from football_research.acquisition.support_round10 import support_features,distribution,_public_record
from football_research.acquisition.joint_returns import entropy


def record(i=0,evsi=.03,acquired=()):
    p=[.4+i/1000,.3,.3-i/1000]
    return {"match_id":str(i),"action_id":"results_view","acquired_action_ids":list(acquired),
        "model_evsi":evsi,"p0":p,"p0_entropy":entropy(p)}


def test_support_flags_do_not_read_outcomes_or_unrelated_payload():
    class Poison:
        def __float__(self): raise AssertionError("Outcome used for support")
        def __str__(self): raise AssertionError("Hidden field read")
    fit=[record(i,.025+i*.0005) for i in range(20)]
    query=record(80,.041)
    clean=support_features(query,fit)
    query["actual_gain"]=Poison(); query["result"]=Poison()
    for r in fit:
        r["actual_gain"]=Poison(); r["outcome"]=Poison()
    assert support_features(query,fit)==clean
    assert clean["support_status"]=="supported_local"
    assert clean["evsi_local_range"]["in_minmax"] is False
    assert clean["low_local_support"] is False


def test_p0_nearest_fit_reference_excludes_same_match_only_for_fit():
    fit=[record(i,.03) for i in range(15)]
    in_sample=support_features(fit[5],fit)
    loo=support_features(fit[5],fit,leave_one_match_out=True)
    assert in_sample["nearest_fit_p0_distance"]==0
    assert loo["nearest_fit_p0_distance"]==pytest.approx(2**.5/1000)
    assert loo["nearest_fit_p0_match_id"]!=fit[5]["match_id"]
    assert loo["local_support"]==15


def test_fixed_range_and_quantile_distinguish_small_sample_tails():
    fit=[record(i,.025+i*.001) for i in range(20)]
    low=support_features(record(99,.025),fit)
    middle=support_features(record(98,.035),fit)
    assert low["evsi_local_range"]["in_minmax"]
    assert not low["evsi_local_range"]["in_p05_p95"]
    assert middle["evsi_local_range"]["in_p05_p95"]
    assert distribution([])["p05"] is None


def test_observed_missing_context_and_bucket_boundary_are_not_merged():
    fit=[record(i,.03,acquired=("xg_view",)) for i in range(15)]
    assert support_features(record(70,.03),fit)["parent_support"]==0
    same=support_features(record(70,.03,acquired=("xg_view",)),fit)
    assert same["local_support"]==15
    boundary=support_features(record(70,.05,acquired=("xg_view",)),fit)
    assert boundary["parent_support"]==15 and boundary["local_support"]==0
    assert boundary["support_status"]=="supported_parent"
    assert boundary["evsi_local_range"]["in_minmax"] is None
    assert boundary["nearest_fit_p0_distance"] is None


def test_duplicate_support_and_invalid_public_probability_rejected():
    fit=[record(i) for i in range(15)]
    with pytest.raises(ValueError,match="one record per match"):
        support_features(record(70),fit+[deepcopy(fit[0])])
    bad=record(70); bad["p0"]=[.5,.3]
    with pytest.raises(ValueError,match="normalized"):
        support_features(bad,fit)


def test_equal_declared_fit_cutoff_allowed_only_with_strictly_earlier_actual_labels():
    row={"match_id":"future","prediction_cutoff":"2025-05-01T00:00:00+00:00","kickoff":"2025-05-02T00:00:00+00:00"}
    meta={"training_cutoff":row["prediction_cutoff"],"latest_training_label_at":"2025-04-30T22:00:00+00:00",
          "version":"fixture","joint_training_matches":100,"base_training_matches":140}
    result=_public_record(row,"outer_validation",0,[.4,.3,.3],[.4,.3,.3],.03,"results_view",[],meta,100)
    assert result["training_cutoff_age_days"]==0
    for invalid in (row["prediction_cutoff"],"2025-05-01T01:00:00+00:00"):
        with pytest.raises(ValueError,match="boundary"):
            _public_record(row,"outer_validation",0,[.4,.3,.3],[.4,.3,.3],.03,"results_view",[],{**meta,"latest_training_label_at":invalid},100)
