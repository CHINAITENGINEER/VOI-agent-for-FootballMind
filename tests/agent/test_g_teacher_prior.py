import json
from pathlib import Path

import pytest

from football_agent.g_teacher_prior import (SyntheticGTeacherPriorAdapter,
                                             load_prior, rank_candidates, score_candidate)


MODEL = Path("var/submission/p9-g-training-bridge-20261010/synthetic-g-model.json")


def _candidate(kind, *, uncertainty, sensitivity, reliability, specificity, redundancy, misdirection, cost):
    return {
        "information_kind": kind,
        "uncertainty": uncertainty,
        "decision_sensitivity": sensitivity,
        "source_reliability": reliability,
        "specificity": specificity,
        "redundancy": redundancy,
        "misdirection_risk": misdirection,
        "query_cost": cost,
    }


def test_prior_ranks_high_sensitivity_lineup_above_h2h():
    model = load_prior(MODEL)
    rows = rank_candidates(model, [
        {"candidate_id": "h2h", **_candidate("historical_h2h", uncertainty=.2, sensitivity=.2,
                                                reliability=.5, specificity=.4, redundancy=.7,
                                                misdirection=.4, cost=.05)},
        {"candidate_id": "lineup", **_candidate("lineup_injury", uncertainty=.9, sensitivity=.9,
                                                   reliability=.8, specificity=.9, redundancy=.1,
                                                   misdirection=.2, cost=.02)},
    ])
    assert rows[0]["candidate_id"] == "lineup"
    assert rows[0]["support_status"] == "SYNTHETIC_PRIOR"
    assert rows[0]["production_ready"] is False


def test_prior_subtracts_cost_once():
    model = load_prior(MODEL)
    candidate = _candidate("lineup_injury", uncertainty=.9, sensitivity=.9, reliability=.8,
                           specificity=.9, redundancy=.1, misdirection=.2, cost=.02)
    scored = score_candidate(model, candidate)
    assert scored["net_value"] == pytest.approx(scored["expected_gain"] - .02)
    assert scored["evidence_origin"] == "synthetic_domain_teacher"


def test_prior_rejects_missing_or_out_of_range_state():
    model = load_prior(MODEL)
    candidate = _candidate("weather", uncertainty=1.4, sensitivity=.2, reliability=.8,
                           specificity=.5, redundancy=.2, misdirection=.2, cost=.02)
    with pytest.raises(ValueError):
        score_candidate(model, candidate)


def test_adapter_loads_bridge_and_keeps_synthetic_provenance_separate():
    bridge = Path("var/submission/p9-g-training-bridge-20261010/g-training-bridge.json")
    adapter = SyntheticGTeacherPriorAdapter.from_path(bridge)
    assert adapter.manifest["container_schema"] == "footballmind-g-training-bridge-v1"
    assert adapter.manifest["synthetic"] is True
    assert adapter.manifest["empirical"] is False
    assert adapter.manifest["selection_mode"] == "prior_ranking_only_empirical_gate_unchanged"


def test_adapter_prior_does_not_promote_unknown_empirical_support():
    adapter = SyntheticGTeacherPriorAdapter.from_path(MODEL)
    candidate = {"candidate_id": "lineup", **_candidate(
        "lineup_injury", uncertainty=.864, sensitivity=.9, reliability=.82,
        specificity=.88, redundancy=.1, misdirection=.2, cost=.024)}
    empirical = {"status": "INELIGIBLE", "expected_signed_gain": None, "net_voi": None,
                 "estimate": {"support_status": "UNKNOWN", "reason": "no_empirical_neighbors"}}
    row = adapter.assess(candidate, empirical_value=empirical)
    assert row["synthetic_prior"]["status"] == "PRIOR_ONLY"
    assert row["synthetic_prior"]["prior_net_value"] == pytest.approx(.0635774320054664)
    assert row["effective"] == {
        "support_status": "UNKNOWN", "expected_signed_gain": None, "net_voi": None,
        "selection_eligible": False, "gate_reason": "empirical_support_unknown"}
    assert row["selection_allowed"] is False
    assert row["provenance"]["synthetic"]["evidence_origin"] == "synthetic_domain_teacher"


def test_adapter_preserves_supported_empirical_value_beside_prior():
    adapter = SyntheticGTeacherPriorAdapter.from_path(MODEL)
    candidate = {"candidate_id": "lineup", **_candidate(
        "lineup_injury", uncertainty=.864, sensitivity=.9, reliability=.82,
        specificity=.88, redundancy=.1, misdirection=.2, cost=.024)}
    empirical = {"status": "COMPUTED", "expected_signed_gain": .14, "net_voi": .11,
                 "estimate": {"support_status": "supported", "evidence_origin": "real_observed"}}
    attached = adapter.attach(candidate, empirical)
    assert attached["expected_signed_gain"] == .14
    assert attached["net_voi"] == .11
    assert attached["synthetic_prior"]["effective"]["selection_eligible"] is True
    assert attached["synthetic_prior"]["synthetic_prior"]["prior_net_value"] == pytest.approx(.0635774320054664)
    assert attached["synthetic_prior"]["provenance"]["empirical"]["evidence_origin"] == "real_observed"


def test_empty_diagnostic_menu_cannot_authorize_selection():
    adapter = SyntheticGTeacherPriorAdapter.from_path(MODEL)
    assert adapter.rank_diagnostic([])["selection_allowed"] is False
