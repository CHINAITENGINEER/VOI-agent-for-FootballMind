from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.g_prior_features import (
    TEACHING_SCHEMA,
    attach_any_prior_features,
    validate_teaching_prior_features,
)
from football_agent.learned_voi_runtime import _actions
from football_agent.synthetic_g_teaching_prior import SyntheticGTeachingPriorAdapter
from football_agent.g_teacher_prior import SyntheticGTeacherPriorAdapter, load_prior


MODEL = Path("var/submission/p9-g-synthetic-teaching-20261010/model.json")
BRIDGE = Path("var/submission/p9-g-training-bridge-20261010/g-training-bridge.json")
OLD_MODEL = BRIDGE.with_name("synthetic-g-model.json")


def test_prior_adapters_reject_other_artifact_families():
    with pytest.raises(ValueError):
        SyntheticGTeacherPriorAdapter.from_path(MODEL)
    with pytest.raises(ValueError):
        load_prior(MODEL)
    for other in (BRIDGE, OLD_MODEL):
        with pytest.raises(ValueError):
            SyntheticGTeachingPriorAdapter.from_path(other)


def test_bridge_envelope_changes_source_digest_but_not_extracted_model(tmp_path):
    payload = json.loads(BRIDGE.read_text(encoding="utf-8"))
    payload["one_match_probe"] = {"note": "different envelope, unchanged synthetic model"}
    path = tmp_path / "bridge.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    original = SyntheticGTeacherPriorAdapter.from_path(BRIDGE)
    changed = SyntheticGTeacherPriorAdapter.from_path(path)
    assert changed.source_sha256 == sha256(path.read_bytes()).hexdigest()
    assert changed.source_sha256 != original.source_sha256
    assert changed.model_sha256 == original.model_sha256
    assert load_prior(path) == load_prior(OLD_MODEL)


def features(action="lineup", scenario="rotation_sensitive", *, cost=.018):
    return {
        "schema": TEACHING_SCHEMA,
        "action": action,
        "scenario": scenario,
        "entropy": .8974526644565144,
        "max_probability": .534700471737682,
        "time_to_kickoff_hours": 24.,
        "source_reliability": .94,
        "event_relevance": .96,
        "tactical_sensitivity": .5,
        "cost_units": cost,
    }


def candidate(action_id="lineup", declared=None):
    return {"candidate_id": action_id,
            "synthetic_prior_features": features(action_id) if declared is None else declared}


def test_new_teaching_adapter_loads_and_keeps_explicit_provenance():
    adapter = SyntheticGTeachingPriorAdapter.from_path(MODEL)
    row = adapter.prior(candidate())
    assert row["status"] == "PRIOR_ONLY"
    assert row["support_status"] == "synthetic_teaching_only"
    assert row["evidence_origin"] == "constructed_mechanism"
    assert row["production_ready"] is False
    assert row["model_schema"] == "synthetic-g-teaching-v1"
    assert row["prior_net_value"] == pytest.approx(row["expected_net_voi"])


def test_missing_features_are_unavailable_and_never_inferred():
    adapter = SyntheticGTeachingPriorAdapter.from_path(MODEL)
    row = adapter.prior({"candidate_id": "lineup", "url": "https://example.com/lineup"})
    assert row["status"] == "UNAVAILABLE"
    assert row["prior_net_value"] is None
    assert row["reason"] == "teaching_prior_features_missing_or_invalid"


def test_unknown_empirical_gate_is_preserved_beside_prior():
    adapter = SyntheticGTeachingPriorAdapter.from_path(MODEL)
    empirical = {"status": "INELIGIBLE", "expected_signed_gain": None,
                 "net_voi": None,
                 "estimate": {"support_status": "UNKNOWN", "reason": "no_neighbors"}}
    attached = adapter.attach(candidate(), empirical)
    assert attached["status"] == "INELIGIBLE"
    assert attached["expected_signed_gain"] is None
    assert attached["net_voi"] is None
    lane = attached["synthetic_prior"]
    assert lane["effective"]["support_status"] == "UNKNOWN"
    assert lane["selection_allowed"] is False
    assert lane["synthetic_prior"]["support_status"] == "synthetic_teaching_only"


def test_supported_empirical_value_is_unchanged_and_prior_only_ranks():
    adapter = SyntheticGTeachingPriorAdapter.from_path(MODEL)
    empirical = {"status": "COMPUTED", "expected_signed_gain": .14,
                 "net_voi": .11, "estimate": {"support_status": "supported"}}
    before = deepcopy(empirical)
    attached = adapter.attach(candidate(), empirical)
    assert empirical == before
    assert attached["expected_signed_gain"] == .14
    assert attached["net_voi"] == .11
    assert attached["synthetic_prior"]["selection_allowed"] is True
    rows = adapter.rank_diagnostic([candidate("weather", features("weather", "rotation_sensitive", cost=.007)),
                                   candidate()], empirical_values={"lineup": empirical, "weather": empirical})["rows"]
    assert rows[0]["candidate_id"] == "lineup"


def test_versioned_features_are_checked_before_runtime_transport():
    declared = {"one": features(), "two": features("weather", "weather_sensitive", cost=.007)}
    actions = [{"action_id": "one", "information_key": "one", "group_ids": [],
                "tool": "read_url", "url": "https://example.com/one", "available": True,
                "cost_status": "simulated", "cost": {"money": .01, "requests": 1}},
               {"action_id": "two", "information_key": "two", "group_ids": [],
                "tool": "read_url", "url": "https://example.com/two", "available": True,
                "cost_status": "simulated", "cost": {"money": .01, "requests": 1}}]
    enriched = attach_any_prior_features(actions, declared)
    assert all(item["synthetic_prior_features"]["schema"] == TEACHING_SCHEMA for item in enriched)
    assert _actions(enriched)[0]["synthetic_prior_features"]["action"] == "lineup"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_teaching_prior_features({**features(), "title": "must not be consumed"})
