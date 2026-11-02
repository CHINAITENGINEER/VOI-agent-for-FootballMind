"""Constructed feature inputs exercise joins, never real evaluation evidence."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest

from football_research.data_adapters.batch_features import (
    FeatureIndex, FeatureJoinError, load_feature_index,
)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture(**changes):
    return {"match_id": "source:match:1", "competition_id": "league:1",
            "home_team_id": "team:1", "away_team_id": "team:2",
            "kickoff": "2025-01-03T12:00:00Z", "information_mode": "historical_reconstruction",
            "home_goals": 2, "away_goals": 1, "result_available_at": "2025-01-04T12:00:00Z",
            "fixture_available_at": "2025-01-01T00:00:00Z", "score_period": "regulation",
            "status": "finished", "input_manifest": ["source:observation:1"], **changes}


def snapshot(**changes):
    manifest = {"match_id": "source:match:1", "prediction_cutoff": "2025-01-02T12:00:00Z",
                "information_mode": "historical_reconstruction", "feature_version": "test-v1",
                "history_match_ids": ["source:match:0"], "history_inputs": [{"match_id": "source:match:0",
                    "kickoff": "2024-12-20T12:00:00Z", "available_at": "2024-12-21T12:00:00Z"}]}
    return {**{k: manifest[k] for k in ("match_id", "prediction_cutoff", "information_mode", "feature_version")},
            "features_available_at": "2025-01-02T11:00:00Z", "input_manifest": manifest,
            "input_manifest_hash": digest(manifest), "features": {"home_elo": 1510., "away_elo": 1490.,
                "home_rolling_xg_for": 1.4, "away_rolling_xg_for": None}, **changes}


def changed_manifest(row, **changes):
    result = deepcopy(row)
    result["input_manifest"].update(changes)
    result["input_manifest_hash"] = digest(result["input_manifest"])
    return result


def join(index, target=None, **changes):
    return index.join(target or fixture(), **{"prediction_cutoff": "2025-01-02T12:00:00Z",
        "information_mode": "historical_reconstruction", "feature_version": "test-v1",
        "required_features": ("home_elo", "away_elo"), **changes})


def test_exact_join_and_supervised_training_assembly():
    index = FeatureIndex.from_rows([snapshot()])
    features = join(index)
    assert features["home_elo"] == 1510.
    assert features["features_available_at"] < features["prediction_cutoff"]
    rows, report = index.assemble_training_rows([fixture()], feature_version="test-v1",
        lead_hours=24, required_features=("home_elo", "away_elo"))
    assert report["accepted_rows"] == 1 and not report["exclusions"]
    assert rows[0]["home_goals"] == 2
    assert rows[0]["feature_input_manifest"]["feature_snapshot_sha256"] == features["snapshot_sha256"]
    assert rows[0]["prediction_cutoff"] == datetime(2025, 1, 2, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("change", [{"prediction_cutoff": "2025-01-02T12:00:01Z"},
    {"information_mode": "operational"}, {"feature_version": "different-v2"}])
def test_no_nearest_cutoff_or_mode_or_version_substitution(change):
    with pytest.raises(FeatureJoinError):
        join(FeatureIndex.from_rows([snapshot()]), **change)


@pytest.mark.parametrize("change,reason", [
    ({"features_available_at": "2025-01-02T12:00:01Z"}, "future_feature_availability"),
    ({"elo_available_at": "2025-01-02T12:00:01Z"}, "future_feature_availability"),
    ({"input_manifest_hash": "broken"}, "feature_manifest_hash_mismatch"),
    ({"feature_version": "wrong-version"}, "manifest_identity_mismatch"),
    ({"features_available_at": None}, "unknown_feature_availability"),
    ({"kickoff": "2025-01-02T11:00:00Z"}, "feature_cutoff_not_prematch"),
])
def test_bad_snapshots_quarantined_before_index(change, reason):
    index = FeatureIndex.from_rows([snapshot(**change)])
    assert index.report()["accepted_snapshots"] == 0
    assert index.report()["rejections"][0]["reason"] == reason


def test_future_append_leaves_earlier_snapshot_and_lineage_identical():
    first = snapshot()
    future = changed_manifest(snapshot(prediction_cutoff="2025-01-03T12:00:00Z",
        features_available_at="2025-01-03T11:00:00Z"), prediction_cutoff="2025-01-03T12:00:00Z")
    poisoned_same_key = snapshot(features_available_at="2025-01-04T00:00:00Z",
        features={"home_elo": 9999., "away_elo": 0.})
    assert join(FeatureIndex.from_rows([first])) == join(FeatureIndex.from_rows([first, future, poisoned_same_key]))


def test_duplicate_is_idempotent_but_conflict_quarantines_key():
    row = snapshot()
    duplicate = FeatureIndex.from_rows([row, deepcopy(row)])
    assert duplicate.report()["identical_duplicates"] == 1
    assert join(duplicate)["home_elo"] == 1510.
    changed = snapshot(features={"home_elo": 1600., "away_elo": 1490.})
    for records in ([row, changed], [changed, row]):
        index = FeatureIndex.from_rows(records)
        assert index.report()["conflicting_keys"] == 1
        with pytest.raises(FeatureJoinError, match="conflicting_duplicate_snapshot"):
            join(index)


def test_ids_never_join_implicitly_and_explicit_mapping_preserves_evidence():
    target = fixture(match_id="canonical:1")
    with pytest.raises(FeatureJoinError, match="missing_exact_feature_snapshot"):
        join(FeatureIndex.from_rows([snapshot()]), target)
    aliases = {"source:match:1": {"target_id": "canonical:1", "evidence": "mapping:verified:1"}}
    output = join(FeatureIndex.from_rows([snapshot()], aliases=aliases), target)
    assert output["match_id"] == "canonical:1"
    assert output["input_manifest"]["id_mapping"][0]["evidence"] == "mapping:verified:1"
    with pytest.raises(ValueError, match="requires target_id and evidence"):
        FeatureIndex(aliases={"source:match:1": "canonical:1"})


def test_wrong_team_id_is_rejected_even_when_match_id_agrees():
    index = FeatureIndex.from_rows([snapshot(home_team_id="team:99")])
    with pytest.raises(FeatureJoinError, match="feature_fixture_identity_mismatch"):
        join(index)


def test_target_and_future_inputs_are_excluded():
    target = changed_manifest(snapshot(), history_match_ids=["source:match:1"])
    future = changed_manifest(snapshot(), history_inputs=[{"match_id": "source:match:2",
        "kickoff": "2025-01-02T10:00:00Z", "available_at": "2025-01-02T13:00:00Z"}])
    index = FeatureIndex.from_rows([target, future])
    assert [r["reason"] for r in index.rejections] == ["target_match_in_feature_history", "future_feature_history"]


@pytest.mark.parametrize("history_key", ["history_inputs", "history_matches"])
@pytest.mark.parametrize("time_key", ["xg_available_at", "elo_available_at"])
def test_distinct_history_xg_and_elo_availability_must_precede_cutoff(history_key, time_key):
    item = {"match_id": "source:match:0", "kickoff": "2025-01-01T00:00:00Z",
            "result_available_at": "2025-01-01T03:00:00Z", time_key: "2025-01-02T12:00:01Z"}
    record = changed_manifest(snapshot(), **{history_key: [item]})
    assert FeatureIndex.from_rows([record]).rejections[0]["reason"] == "future_feature_history"
    item[time_key] = "2025-01-02T12:00:00Z"
    record = changed_manifest(snapshot(), **{history_key: [item]})
    assert FeatureIndex.from_rows([record], strict=True).report()["accepted_snapshots"] == 1


def test_manifest_maximum_input_availability_rejected_even_with_earlier_row_time():
    record = changed_manifest(snapshot(), max_input_available_at="2025-01-02T12:00:01Z")
    assert FeatureIndex.from_rows([record]).rejections[0]["reason"] == "future_feature_history"
    record = changed_manifest(snapshot(), max_input_available_at="2025-01-02T12:00:00Z")
    assert FeatureIndex.from_rows([record], strict=True).report()["accepted_snapshots"] == 1


def test_both_history_lists_checked_when_export_contains_both():
    record = changed_manifest(snapshot(), history_matches=[], history_inputs=[{
        "match_id": "source:match:0", "xg_available_at": "2025-01-02T12:00:01Z"}])
    assert FeatureIndex.from_rows([record]).rejections[0]["reason"] == "future_feature_history"


def test_observed_after_cutoff_allowed_only_in_reconstruction():
    record = changed_manifest(snapshot(), history_inputs=[{"match_id": "old:1",
        "available_at": "2024-01-01T00:00:00Z", "observed_at": "2026-01-01T00:00:00Z"}])
    assert FeatureIndex.from_rows([record]).report()["accepted_snapshots"] == 1
    record["information_mode"] = "operational"
    record = changed_manifest(record, information_mode="operational")
    assert FeatureIndex.from_rows([record]).rejections[0]["reason"] == "future_operational_observation"


def test_missing_field_and_null_value_are_distinct_and_imputation_is_opt_in():
    index = FeatureIndex.from_rows([snapshot(features={"home_elo": None, "away_elo": 1490.})])
    with pytest.raises(FeatureJoinError, match="missing_required_feature_values"):
        join(index)
    assert join(index, allow_missing_values=True)["input_manifest"]["missing_values"] == ["home_elo"]
    with pytest.raises(FeatureJoinError, match="missing_required_feature_fields"):
        join(index, required_features=("home_form_points",), allow_missing_values=True)
    rows, report = index.assemble_training_rows([fixture()], feature_version="test-v1",
        required_features=("home_elo", "away_elo"), allow_missing_values=True)
    assert rows[0]["home_elo"] is None and report["missing_values"] == {"home_elo": 1}


def test_forbidden_fields_isolated_and_historical_xg_allowed():
    row = snapshot()
    row["features"].update(home_goals=8, home_xg=7.2, forecast=.9, final_rank=1, home_season_total_xg=88.)
    out = join(FeatureIndex.from_rows([row]))
    assert out["home_rolling_xg_for"] == 1.4
    for key in ("home_goals", "home_xg", "forecast", "final_rank", "home_season_total_xg"):
        assert key not in out
        assert key in out["input_manifest"]["excluded_fields"]
        with pytest.raises(ValueError, match="Disallowed"):
            FeatureIndex(allowed_features=[key])


def test_consumed_rows_are_copied_and_caller_mutation_does_not_change_index():
    index = FeatureIndex.from_rows([snapshot()])
    output = join(index)
    output["home_elo"] = 0
    output["input_manifest"]["source_manifest"]["history_match_ids"].append("source:match:1")
    assert join(index)["home_elo"] == 1510.
    assert join(index)["input_manifest"]["source_manifest"]["history_match_ids"] == ["source:match:0"]


def test_jsonl_and_completed_directory_integrity(tmp_path):
    file = tmp_path / "features.jsonl"
    file.write_text(json.dumps(snapshot()) + "\n", encoding="utf-8")
    assert load_feature_index(file).report()["accepted_snapshots"] == 1
    with pytest.raises(ValueError, match="missing _SUCCESS"):
        load_feature_index(tmp_path)
    (tmp_path / "_SUCCESS").write_text("complete", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "features.jsonl": sha256(file.read_bytes()).hexdigest()}}), encoding="utf-8")
    assert join(load_feature_index(tmp_path))["home_elo"] == 1510.
    file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_feature_index(tmp_path)


def test_existing_platform_parquet_projection(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    manifest = {"match_id": "source:match:1", "prediction_cutoff": "2025-01-02T12:00:00Z",
        "feature_version": "test-v1", "information_mode": "historical_reconstruction",
        "availability_basis": "trusted_provider_snapshot_plus_assumed_delay",
        "history_version_ids": ["history-version:1"], "parameters": {"k": 20}}
    features = {"elo": {"home": 1510., "away": 1490., "parameters": {"k": 20},
        "input_version_ids": ["history-version:1"]}, "home_form": {"points": 9, "rest_days": 7.}}
    row = {**{key: manifest[key] for key in ("match_id", "prediction_cutoff", "feature_version", "information_mode")},
        "manifest_json": json.dumps(manifest), "features_json": json.dumps(features),
        "input_manifest_hash": digest(manifest)}
    file = tmp_path / "feature.parquet"
    pq.write_table(pa.Table.from_pylist([row]), file)
    output = join(load_feature_index(file, strict=True))
    assert output["home_form_points"] == 9 and output["home_rest_days"] == 7.


def test_optional_snapshot_hash_detects_feature_value_tampering():
    row = snapshot()
    row["snapshot_sha256"] = digest(row)
    assert FeatureIndex.from_rows([row]).report()["accepted_snapshots"] == 1
    row["features"]["home_elo"] = 2000.
    assert FeatureIndex.from_rows([row]).rejections[0]["reason"] == "snapshot_content_hash_mismatch"


def test_completed_platform_bundle_reuses_history_integrity_verification(tmp_path):
    from football_research.data_adapters.platform_history import EXPORT_SCHEMA
    files = {"matches": (json.dumps(fixture()) + "\n").encode(),
             "features": (json.dumps(snapshot()) + "\n").encode()}
    manifest = {"schema": EXPORT_SCHEMA, "dataset_version": "test-only-v1", "files": {}}
    for kind, payload in files.items():
        (tmp_path / (kind + ".jsonl")).write_bytes(payload)
        manifest["files"][kind] = {"path": kind + ".jsonl", "row_count": 1,
                                   "sha256": sha256(payload).hexdigest()}
    encoded = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256(encoded).hexdigest()}))
    assert join(load_feature_index(tmp_path))["home_elo"] == 1510.
    (tmp_path / "features.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_feature_index(tmp_path)


def test_training_join_reports_missing_rows_without_label_or_feature_substitution():
    rows, report = FeatureIndex.from_rows([snapshot()]).assemble_training_rows(
        [fixture(), fixture(match_id="source:match:unknown")], feature_version="test-v1")
    assert len(rows) == 1 and report["excluded_rows"] == 1
    assert report["exclusions"][0]["reason"] == "missing_exact_feature_snapshot"


def test_training_conflicting_labels_are_quarantined_not_silently_deduplicated():
    rows, report = FeatureIndex.from_rows([snapshot()]).assemble_training_rows(
        [fixture(), fixture(home_goals=0)], feature_version="test-v1")
    assert rows == []
    assert report["exclusions"][0]["reason"] == "conflicting_training_match_versions"


@pytest.mark.parametrize("history,reason", [
    ({"match_id": "source:match:0", "xg_available_at": "2030-01-01T00:00:00Z"}, "future_feature_history"),
    ({"match_id": "source:match:1", "xg_available_at": "2025-01-01T00:00:00Z"}, "target_match_in_feature_history"),
])
def test_platform_separate_xg_lineage_is_validated(history, reason):
    row = snapshot()
    row["input_manifest"]["xg_inputs"] = [history]
    row["input_manifest_hash"] = digest(row["input_manifest"])
    assert FeatureIndex.from_rows([row]).rejections[0]["reason"] == reason
