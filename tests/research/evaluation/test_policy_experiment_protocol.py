from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from football_research.evaluation.policy_experiment_protocol import (
    audit_episode_schema,
    audit_holdout_manifest,
    audit_parameter_freeze,
    audit_policy_experiment,
    manifest_hash,
)


def episode(**overrides):
    row = {
        "dataset_version": "fixture-v1",
        "fold": 0,
        "match_id": "m1",
        "competition_id": "league-1",
        "league": "League One",
        "kickoff": "2025-01-10T12:00:00Z",
        "prediction_cutoff": "2025-01-10T11:00:00Z",
        "lead_hours": 1.0,
        "fixture_available_at": "2025-01-01T00:00:00Z",
        "strategy": "voi-gate",
        "strategy_version": "voi-gate-v1",
        "candidate_set_version": "candidate-v1",
        "budget_money": 0.02,
        "seed": 7,
        "source_id": "spftll",
        "source_version": "2025-01-01",
        "feature_manifest_hash": manifest_hash({"feature_version": "v1"}),
        "policy_context_hash": manifest_hash({"competition_id": "league-1", "rank_known": True}),
        "policy_parameter_hash": manifest_hash({"parameters": {"version": "v1"}, "thresholds": {"stop": 0.02}}),
        "evaluator_sidecar_ref": "eval/m1.json",
        "final_probabilities": [0.5, 0.3, 0.2],
        "cost": {"money": 0.01, "requests": 1.0},
        "policy_visible_context": {"competition_id": "league-1", "rank_known": True},
    }
    row.update(overrides)
    return row


def frozen_parameters(parameters=None):
    payload = {"parameters": parameters or {"version": "v1"}, "thresholds": {"stop": 0.02}}
    return {
        **payload, "parameter_hash": manifest_hash(payload),
        "fit_partition": "train", "selection_partition": "validation", "evaluation_partition": "test",
        "frozen_at": "2025-01-09T00:00:00Z", "frozen": True,
    }


def independent_run():
    """Synthetic protocol fixture only; never presented as football evidence."""
    manifest = {
        "dataset_version": "fixture-v1", "fold": 0, "candidate_set_version": "candidate-v1",
        "strategy_versions": {"voi-gate": "voi-gate-v1"}, "budgets": [0.02],
        "cost_interpretation": "simulated currency, no real requests",
        "feature_manifest": {"feature_version": "v1"},
        "partitions": {"train": ["a"], "validation": ["b"], "test": ["m1"]},
        "prior_calibration_match_ids": ["a", "b"],
        "prior_policy_search_match_ids": [], "declared_independent": True,
    }
    parameters = {name: deepcopy(manifest[name]) for name in (
        "dataset_version", "fold", "candidate_set_version", "strategy_versions", "budgets", "cost_interpretation",
    )}
    parameters["feature_manifest_hash"] = manifest_hash(manifest["feature_manifest"])
    parameters["holdout_manifest_hash"] = manifest_hash({name: manifest[name] for name in (
        "partitions", "prior_calibration_match_ids", "prior_policy_search_match_ids", "declared_independent",
    )})
    manifest["policy_freeze"] = frozen_parameters(parameters)
    return [episode(policy_parameter_hash=manifest["policy_freeze"]["parameter_hash"])], manifest


def test_complete_episode_schema_passes_without_derivation():
    report = audit_episode_schema([episode()])
    assert report["status"] == "PASS"
    assert report["matches"] == 1
    assert report["missing_field_counts"] == {}
    assert report["invalid_cost_receipt_rows"] == 0


def test_missing_metadata_is_a_schema_gate():
    row = episode()
    for field in ("lead_hours", "league", "strategy_version", "source_version", "policy_context_hash"):
        row.pop(field)
    report = audit_episode_schema([row])
    assert report["status"] == "SCHEMA_REVIEW_REQUIRED"
    assert set(report["missing_field_counts"]) >= {
        "lead_hours", "league", "strategy_version", "source_version", "policy_context_hash",
    }


def test_public_context_cannot_contain_evaluator_label_fields():
    report = audit_episode_schema([episode(policy_visible_context={"outcome": "home"})])
    assert report["status"] == "SCHEMA_REVIEW_REQUIRED"
    assert report["public_context_label_paths"] == 1
    assert report["issue_counts"]["PUBLIC_CONTEXT_LABEL_FIELD"] == 1


def test_holdout_requires_disjoint_test_and_prior_manifests():
    report = audit_holdout_manifest(
        {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        prior_calibration_match_ids=["a", "d"],
        prior_policy_search_match_ids=[],
        episode_match_ids=["a", "b", "c", "d"],
        declared_independent=True,
    )
    assert report["status"] == "HOLDOUT_REVIEW_REQUIRED"
    assert report["test_overlap_prior_count"] == 1
    assert report["episode_outside_manifest_count"] == 0


def test_independent_holdout_and_parameter_freeze_can_pass():
    split = audit_holdout_manifest(
        {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        prior_calibration_match_ids=["a", "b", "c"],
        prior_policy_search_match_ids=[],
        episode_match_ids=["a", "b", "c", "d"],
        declared_independent=True,
    )
    freeze = audit_parameter_freeze(
        frozen_parameters(),
        earliest_test_prediction_cutoff="2025-01-10T11:00:00Z",
    )
    assert split["status"] == "PASS"
    assert freeze["status"] == "PASS"


def test_manifest_hash_is_canonical_and_rejects_nan():
    first = manifest_hash({"b": 2, "a": [1, 2]})
    second = manifest_hash({"a": [1, 2], "b": 2})
    assert first == second
    assert len(first) == 64
    with pytest.raises(ValueError):
        manifest_hash({"threshold": float("nan")})


def test_empty_episodes_are_not_a_successful_experiment():
    report = audit_episode_schema([])
    assert report["status"] == "SCHEMA_REVIEW_REQUIRED"
    assert report["issue_counts"]["EMPTY_EPISODES"] == 1


@pytest.mark.parametrize("field,value", [
    ("fold", True), ("fold", -1), ("seed", []), ("seed", 1.5),
    ("league", " "), ("source_version", 12), ("strategy_version", ""),
    ("feature_manifest_hash", "fake"), ("policy_context_hash", "0" * 64),
    ("lead_hours", 24), ("prediction_cutoff", "2025-01-10T11:00:00"),
    ("fixture_available_at", "2025-01-10T11:01:00Z"),
    ("final_probabilities", [0.5, 0.3, 0.3]),
    ("cost", {"money": 0.03, "requests": 1}),
    ("cost", {"money": 0.01, "requests": 1.5}),
    ("cost", {"money": float("nan"), "requests": 1}),
])
def test_bad_episode_metadata_fails_closed(field, value):
    assert audit_episode_schema([episode(**{field: value})])["status"] == "SCHEMA_REVIEW_REQUIRED"


def test_duplicate_is_detected_even_with_missing_lead_metadata():
    row = episode()
    row.pop("lead_hours")
    report = audit_episode_schema([row, deepcopy(row)])
    assert report["duplicate_identity_count"] == 1


def test_late_nested_evidence_is_rejected_even_with_valid_context_hash():
    context = {"evidence": [{"available_at": "2025-01-10T11:01:00Z"}]}
    report = audit_episode_schema([episode(policy_visible_context=context, policy_context_hash=manifest_hash(context))])
    assert report["issue_counts"]["PUBLIC_EVIDENCE_AFTER_CUTOFF"] == 1


@pytest.mark.parametrize("partitions", [
    {}, {"train": [], "validation": [], "test": []},
    {"train": ["a"], "validation": ["b"], "test": "m1"},
    {"train": ["a"], "validation": ["b"], "test": ["m1", "m1"]},
    {"train": ["m1"], "validation": ["b"], "test": ["m1"]},
    {"train": ["a"], "validation": ["b"], "test": [" " ]},
])
def test_invalid_or_empty_partitions_are_not_a_holdout(partitions):
    report = audit_holdout_manifest(partitions, prior_calibration_match_ids=[],
                                    prior_policy_search_match_ids=[], declared_independent=True)
    assert report["status"] == "HOLDOUT_REVIEW_REQUIRED"


def test_omitted_prior_inventory_is_not_treated_as_known_empty():
    report = audit_holdout_manifest({"train": ["a"], "validation": ["b"], "test": ["m1"]}, declared_independent=True)
    assert sum(i["code"] == "INVALID_ID_INVENTORY" for i in report["issues"]) == 2


def test_truthy_string_does_not_declare_independence():
    report = audit_holdout_manifest({"train": ["a"], "validation": ["b"], "test": ["m1"]},
                                    prior_calibration_match_ids=[], prior_policy_search_match_ids=[], declared_independent="false")
    assert report["declared_independent"] is False
    assert report["status"] == "HOLDOUT_REVIEW_REQUIRED"


def test_prior_policy_search_is_distinct_from_supervised_training():
    rows, manifest = independent_run()
    manifest["prior_policy_search_match_ids"] = ["m1"]
    report = audit_policy_experiment(rows, manifest)
    assert report["holdout"]["test_overlap_prior_count"] == 0
    assert report["holdout"]["test_overlap_prior_policy_search_count"] == 1
    assert report["protocol_consistent"] is False


@pytest.mark.parametrize("field,value", [
    ("frozen", "true"), ("frozen_at", "not-a-date"),
    ("frozen_at", "2025-01-10T11:00:00Z"), ("frozen_at", "2025-01-11T00:00:00Z"),
    ("thresholds", {}), ("thresholds", {"stop": float("nan")}),
    ("thresholds", {"stop": 0.05}), ("parameters", {}),
    ("parameters", {"version": "changed"}), ("parameter_hash", "sha256:abc"),
    ("selection_partition", "test"),
])
def test_invalid_or_mutated_freeze_is_rejected(field, value):
    frozen = frozen_parameters()
    frozen[field] = value
    report = audit_parameter_freeze(frozen, earliest_test_prediction_cutoff="2025-01-10T11:00:00Z")
    assert report["status"] == "FREEZE_REVIEW_REQUIRED"


def test_freeze_without_test_boundary_is_not_verified():
    report = audit_parameter_freeze(frozen_parameters())
    assert any(i["code"] == "INVALID_TEST_BOUNDARY" for i in report["issues"])


def test_complete_run_is_consistent_but_not_independent_confirmation():
    rows, manifest = independent_run()
    report = audit_policy_experiment(rows, manifest)
    assert report["status"] == "PROTOCOL_CONSISTENT"
    assert report["errors"] == 0
    assert report["independent_confirmation"] is False


@pytest.mark.parametrize("field,value", [
    ("fold", 1), ("dataset_version", "other"), ("strategy_version", "other"),
    ("candidate_set_version", "other"), ("feature_manifest_hash", "0" * 64),
    ("budget_money", 0.03), ("match_id", "a"), ("policy_parameter_hash", "0" * 64),
])
def test_episode_must_match_frozen_run(field, value):
    rows, manifest = independent_run()
    rows[0][field] = value
    report = audit_policy_experiment(rows, manifest)
    assert report["protocol_consistent"] is False
    assert report["binding_issues"]


@pytest.mark.parametrize("field,value", [
    ("budgets", [0.02, 0.03]), ("cost_interpretation", "real USD"),
    ("feature_manifest", {"feature_version": "v2"}),
    ("partitions", {"train": ["a", "new"], "validation": ["b"], "test": ["m1"]}),
])
def test_changed_run_manifest_invalidates_freeze_binding(field, value):
    rows, manifest = independent_run()
    manifest[field] = value
    report = audit_policy_experiment(rows, manifest)
    assert any(i["code"] == "FROZEN_RUN_BINDING_MISMATCH" for i in report["binding_issues"])


def test_missing_test_matches_are_reported():
    rows, manifest = independent_run()
    manifest["partitions"]["test"].append("m2")
    report = audit_policy_experiment(rows, manifest)
    assert any(i["code"] == "TEST_EPISODE_COVERAGE_MISMATCH" for i in report["binding_issues"])


def test_empty_run_and_manifest_are_rejected():
    report = audit_policy_experiment([], {})
    assert report["protocol_consistent"] is False
    assert report["errors"] > 0


@pytest.mark.parametrize("valid", [True, False])
def test_cli_exit_code_matches_protocol_result(tmp_path, valid):
    rows, manifest = independent_run()
    if not valid:
        rows = []
    episode_path = tmp_path / "episodes.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "audit.json"
    episode_path.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run([
        sys.executable, str(root / "tools/audit_policy_experiment_protocol.py"),
        "--episodes", str(episode_path), "--manifest", str(manifest_path), "--output", str(output_path),
    ], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == (0 if valid else 1), completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["protocol_consistent"] is valid
