from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_research.acquisition.prospective_quality_g import NUMERIC_NAMES, QUERY_VERSION, _digest
from football_research.acquisition.prospective_quality_g_bayesian import (
    PRIOR_SCHEMA, VERSION, BayesianQualityGShadow, fit_shadow,
)


def _query(pipeline="entity-market-f:test", action="read_url"):
    body = {
        "schema": QUERY_VERSION,
        "prediction_pipeline_version": pipeline,
        "predictor_model_version": "f:test",
        "action_id": action,
        "source_match_id": "source-1",
        "numeric_names": NUMERIC_NAMES,
        "feature_origin": "derived_from_saved_pre_io_public_menu_not_online_g_evaluation",
        "context": {
            "texts": ["home preview"], "labels": ["lineup"],
            "numeric": [0.2] * len(NUMERIC_NAMES), "sources": ["example.test"],
            "channel": "read_url|example.test", "mask": [True, True],
            "decision_at": "2026-10-10T10:00:00+00:00",
            "kickoff_at": "2026-10-10T12:00:00+00:00", "date": "2026-10-10",
        },
    }
    body["query_sha256"] = _digest(body)
    return body


def _rows():
    rows = []
    for index, (partition, gain) in enumerate(
            [("train", .20), ("train", .10), ("calibration", .05), ("validation", -.04), ("test", .08)]):
        rows.append({"sample_id": f"a-{index}", "physical_match_id": f"m-{index}",
                     "partition": partition, "target_status": "eligible", "gain": gain,
                     "query": _query()})
    return rows


def test_small_n_posterior_is_shadow_only_and_group_gated():
    model, report = fit_shadow({"records": _rows()})
    assert model is not None
    assert report["production_ready"] is False
    assert report["selection_allowed"] is False
    assert report["production_gate"]["passed"] is False
    estimate = model.estimate(_query())
    assert {"posterior_mean", "posterior_std", "posterior_lcb"} <= set(estimate)
    ranked = model.rank_shadow([{"action_id": "read_url", "query": _query(), "cost_units": .01}])
    assert ranked["selection_allowed"] is False
    assert ranked["production_ready"] is False
    assert ranked["candidates"][0]["posterior_std"] >= 0


def test_synthetic_prior_is_explicitly_wide_and_not_a_label():
    prior = {"schema": PRIOR_SCHEMA, "prior_mean": [1.] * (1 + len(NUMERIC_NAMES) + 96),
             "production_ready": False}
    model, _ = fit_shadow({"records": _rows()}, synthetic_prior=prior)
    assert model.metadata["prior"]["synthetic_is_label"] is False
    assert model.metadata["prior"]["schema"] == PRIOR_SCHEMA
    assert model.metadata["selection_authority"] == "empirical_g_only"


def test_synthetic_or_missing_partition_never_becomes_real_label():
    rows = _rows()
    rows[0]["synthetic"] = True
    model, report = fit_shadow({"records": rows})
    assert model is None
    assert "Synthetic" in report["reason"]
    rows = _rows()
    rows[0].pop("partition")
    model, report = fit_shadow({"records": rows})
    assert model is None
    assert report["fit_status"] == "waiting_for_four_way_group_split"


def test_model_round_trip_preserves_shadow_contract(tmp_path):
    model, _ = fit_shadow({"records": _rows()})
    path = tmp_path / "shadow.json"
    model.save(path)
    loaded = BayesianQualityGShadow.load(path)
    assert loaded.model_version == model.model_version
    assert loaded.estimate(_query())["selection_allowed"] is False
