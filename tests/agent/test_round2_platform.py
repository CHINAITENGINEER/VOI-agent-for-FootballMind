"""Constructed completed exports verify the Agent's platform boundary.

These fixtures are labeled constructed_test and are separate from real platform
acceptance artifacts. Real fitted frequency models are reused where stated.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_agent.adapters import digest
from football_agent.engine import Workflow
from football_agent.model_catalog import ModelCatalog
from football_agent.platform_tools import PlatformTools
from football_agent.types import Budget
from football_research.baselines.api import PredictionService
from football_research.data_adapters.platform_history import EXPORT_SCHEMA


CUTOFF = "2026-09-20T11:25:00Z"
MATCH = "constructed:round2-target"


def match_row(**changes):
    return {"match_id": MATCH, "competition_id": "spftll:competition:36", "season": "2026",
            "home_team_id": "constructed:home", "away_team_id": "constructed:away",
            "home_team_name": "Constructed home", "away_team_name": "Constructed away",
            "competition_name": "Constructed league", "kickoff": "2026-09-20T13:00:00Z",
            "status": "scheduled", "score_period": "regulation",
            "input_manifest": ["constructed-observation:fixture"], **changes}


def feature_row(**changes):
    manifest = {"match_id": MATCH, "prediction_cutoff": CUTOFF, "information_mode": "operational",
                "feature_version": "constructed-feature-v1", "history_match_ids": ["constructed:past"],
                "history_inputs": [{"match_id": "constructed:past", "kickoff": "2026-09-10T13:00:00Z",
                    "available_at": "2026-09-10T16:00:00Z", "observed_at": "2026-09-10T16:00:00Z"}]}
    return {**{k: manifest[k] for k in ("match_id", "prediction_cutoff", "information_mode", "feature_version")},
            "features_available_at": "2026-09-20T11:20:00Z", "input_manifest": manifest,
            "input_manifest_hash": digest(manifest), "home_team_id": "constructed:home",
            "away_team_id": "constructed:away", "competition_id": "spftll:competition:36",
            "features": {"home_elo": 1510., "away_elo": 1490.}, **changes}


def change_manifest(row, **changes):
    row = deepcopy(row)
    row["input_manifest"].update(changes)
    row["input_manifest_hash"] = digest(row["input_manifest"])
    return row


def export_bundle(path, *, row=None, features=None, mode="operational", observed="2026-09-20T11:00:00Z"):
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": EXPORT_SCHEMA, "dataset_version": "constructed-platform-export-v1",
                "source": "constructed_test", "source_observed_at": observed, "information_mode": mode,
                "source_evidence": [{"observation_id": "constructed-raw", "sha256": "a" * 64}],
                "assumptions": {"fixture_lead_hours": 168, "result_delay_hours": 3, "xg_delay_hours": 24,
                                "kickoff_timezone_policy": "constructed_UTC"}, "files": {}}
    payloads = {"matches": [row or match_row()]}
    if features is not None:
        payloads["features"] = features
    for kind, rows in payloads.items():
        payload = "".join(json.dumps(item) + "\n" for item in rows).encode()
        (path / f"{kind}.jsonl").write_bytes(payload)
        manifest["files"][kind] = {"path": f"{kind}.jsonl", "sha256": sha256(payload).hexdigest(), "row_count": len(rows)}
    encoded = json.dumps(manifest).encode()
    (path / "manifest.json").write_bytes(encoded)
    (path / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256(encoded).hexdigest()}))
    return path


def elo_catalog(path, mode="operational"):
    service = PredictionService.fixed_elo_mapping(as_of="2026-09-20T10:00:00Z",
                                                  dataset_version="constructed-elo-map-v1", information_mode=mode)
    model_path = service.save(path / "constructed-elo.json")
    registry_path = path / "registry.json"
    registry_path.write_text(json.dumps({"models": [{"model_id": "constructed-elo", "model_name": "elo",
        "model_artifact": str(model_path), "artifact_sha256": service.to_artifact()["artifact_sha256"],
        "information_mode": mode, "competition_id": "spftll:competition:36"}]}))
    return ModelCatalog(registry_path), service


def query_inputs(**changes):
    return {"match_id": MATCH, "prediction_cutoff": CUTOFF, **changes}


def test_completed_export_fixture_evidence_and_real_model_without_odds(tmp_path):
    path = export_bundle(tmp_path / "export")
    tools = PlatformTools(export_path=path, model_id="frequency-england")
    inputs = query_inputs()
    fixture = tools.execute("query_matches", inputs).data[0]
    evidence = tools.execute("read_evidence", inputs)
    assert evidence.status == "OK"
    assert evidence.data[0]["raw_ref"].endswith("manifest.json")
    assert evidence.data[0]["source_evidence"][0]["observation_id"] == "constructed-raw"
    features = tools.execute("get_features", {**inputs, "match": fixture, "evidence": evidence.data})
    assert features.status == "OK", features.error
    assert features.data["platform_export_sha256"] == tools.export.manifest_sha256
    assert "odds_home" not in features.data
    result = tools.execute("predict", {**inputs, "match": fixture, "features": features.data})
    direct = PredictionService.load(tools.model_artifact).predict(fixture, prediction_cutoff=CUTOFF)
    assert result.status == "OK", result.error
    assert result.data == direct.model_dump(mode="json")
    workflow = Workflow(tools, store_dir=tmp_path / "runs")
    run = workflow.start(MATCH, CUTOFF, Budget(max_queries=0))
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "STOPPED" and run["prediction"] == result.data
    assert workflow.store.verify(run["run_id"])["status"] == "OK"


def test_market_without_verified_quotes_stays_no_data(tmp_path):
    tools = PlatformTools(export_path=export_bundle(tmp_path))
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(), "match": fixture, "evidence": []})
    assert result.status == "NO_DATA"


def test_missing_completion_marker_is_rejected(tmp_path):
    path = export_bundle(tmp_path)
    (path / "COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        PlatformTools(export_path=path)


@pytest.mark.parametrize("name", ["manifest.json", "matches.jsonl", "features.jsonl"])
def test_wrong_export_hash_is_rejected(tmp_path, name):
    path = export_bundle(tmp_path, features=[feature_row()])
    with (path / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        PlatformTools(export_path=path)


def test_export_mutation_after_construction_is_reported_as_tool_failure(tmp_path):
    path = export_bundle(tmp_path)
    tools = PlatformTools(export_path=path)
    (path / "matches.jsonl").write_text("{}\n")
    result = tools.execute("query_matches", query_inputs())
    assert result.status == "FAILED" and result.error["reason"] == "PLATFORM_EXPORT_INVALID"


def test_wrong_export_or_call_mode_is_rejected(tmp_path):
    path = export_bundle(tmp_path)
    with pytest.raises(ValueError, match="information mode"):
        PlatformTools(export_path=path, information_mode="historical_reconstruction")
    result = PlatformTools(export_path=path).execute("query_matches", query_inputs(information_mode="historical_reconstruction"))
    assert result.status == "FAILED" and result.error == "INFORMATION_MODE_MISMATCH"


def test_live_historical_mode_is_rejected_before_execution(tmp_path):
    with pytest.raises(ValueError, match="operational"):
        PlatformTools(time_mode="live", information_mode="historical_reconstruction")


def test_future_observation_and_wrong_match_are_not_visible(tmp_path):
    tools = PlatformTools(export_path=export_bundle(tmp_path, observed="2026-09-20T11:26:00Z"))
    assert tools.execute("query_matches", query_inputs()).status == "NO_DATA"
    assert tools.execute("read_evidence", query_inputs()).status == "NO_DATA"
    tools = PlatformTools(export_path=export_bundle(tmp_path, observed="2026-09-20T11:00:00Z"))
    assert tools.execute("query_matches", query_inputs(match_id="constructed:wrong")).status == "NO_DATA"


def test_target_result_xg_and_forecast_are_excluded_from_fixture_and_evidence(tmp_path):
    row = match_row(home_goals=9, away_goals=0, home_xg=9.9, away_xg=.1, forecast={"home": .99}, season_totals={"xg": 80.})
    tools = PlatformTools(export_path=export_bundle(tmp_path, row=row))
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    evidence = tools.execute("read_evidence", query_inputs()).data[0]
    forbidden = {"home_goals", "away_goals", "home_xg", "away_xg", "forecast", "season_totals", "result_available_at"}
    assert not forbidden.intersection(fixture)
    assert not forbidden.intersection(evidence["content"])


@pytest.mark.parametrize("status", ["postponed", "cancelled", "suspended"])
def test_inactive_schedule_is_not_relabelled_scheduled(tmp_path, status):
    tools = PlatformTools(export_path=export_bundle(tmp_path, row=match_row(status=status)))
    assert tools.execute("query_matches", query_inputs()).status == "NO_DATA"


def test_strict_feature_join_and_prediction_equal_service(tmp_path):
    catalog, service = elo_catalog(tmp_path)
    path = export_bundle(tmp_path / "export", features=[feature_row()])
    tools = PlatformTools(export_path=path, model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(), "match": fixture})
    assert result.status == "OK", result.error
    assert result.data["home_elo"] == 1510.
    assert result.data["feature_version"] == "constructed-feature-v1"
    prediction = tools.execute("predict", {**query_inputs(), "match": fixture, "features": result.data})
    assert prediction.status == "OK", prediction.error
    assert prediction.data == service.predict(fixture, prediction_cutoff=CUTOFF, features=result.data).model_dump(mode="json")


def test_feature_projection_excludes_target_fields_and_retains_exclusion_evidence(tmp_path):
    catalog, _ = elo_catalog(tmp_path)
    snapshot = feature_row()
    forbidden = {"home_goals": 9, "away_goals": 0, "home_xg": 9.1, "forecast": .99, "home_season_total_xg": 90.}
    snapshot["features"].update(forbidden)
    tools = PlatformTools(export_path=export_bundle(tmp_path / "export", features=[snapshot]),
                          model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(), "match": fixture})
    assert result.status == "OK", result.error
    assert not set(forbidden).intersection(result.data)
    assert set(forbidden).issubset(result.data["input_manifest"]["excluded_fields"])


@pytest.mark.parametrize("features,reason", [({"home_elo": 1510.}, "missing_required_feature_fields"),
    ({"home_elo": 1510., "away_elo": None}, "missing_required_feature_values")])
def test_required_feature_gaps_remain_explicit(tmp_path, features, reason):
    catalog, _ = elo_catalog(tmp_path)
    tools = PlatformTools(export_path=export_bundle(tmp_path / "export", features=[feature_row(features=features)]),
                          model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(), "match": fixture})
    assert result.status == "NO_DATA" and reason in result.error


@pytest.mark.parametrize("feature_change,query_change,reason", [
    ({}, {"prediction_cutoff": "2026-09-20T11:25:01Z"}, "missing_exact_feature_snapshot"),
    ({"match_id": "constructed:other"}, {}, "missing_exact_feature_snapshot"),
    ({"information_mode": "historical_reconstruction"}, {}, "missing_exact_feature_snapshot"),
    ({"home_team_id": "constructed:wrong-team"}, {}, "feature_fixture_identity_mismatch"),
])
def test_exact_feature_join_rejects_cutoff_mode_match_and_team_mismatch(tmp_path, feature_change, query_change, reason):
    catalog, _ = elo_catalog(tmp_path)
    snapshot = feature_row(**feature_change)
    manifest_change = {k: v for k, v in feature_change.items() if k in snapshot["input_manifest"]}
    snapshot = change_manifest(snapshot, **manifest_change)
    tools = PlatformTools(export_path=export_bundle(tmp_path / "export", features=[snapshot]),
                          model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(**query_change), "match": fixture})
    assert result.status == "NO_DATA" and reason in result.error


@pytest.mark.parametrize("mutation,reason", [
    (lambda row: {**row, "features_available_at": "2026-09-20T11:26:00Z"}, "future_feature_availability"),
    (lambda row: {**row, "input_manifest_hash": "wrong"}, "feature_manifest_hash_mismatch"),
    (lambda row: change_manifest(row, history_match_ids=[MATCH]), "target_match_in_feature_history"),
    (lambda row: change_manifest(row, history_inputs=[{"match_id": "constructed:past", "available_at": "2026-09-20T11:26:00Z"}]), "future_feature_history"),
])
def test_bad_feature_input_is_rejected_by_strict_export_index(tmp_path, mutation, reason):
    path = export_bundle(tmp_path, features=[mutation(feature_row())])
    with pytest.raises(ValueError, match=reason):
        PlatformTools(export_path=path)


def test_ambiguous_feature_versions_are_not_silently_selected(tmp_path):
    catalog, _ = elo_catalog(tmp_path)
    newer = change_manifest(feature_row(feature_version="constructed-feature-v2"), feature_version="constructed-feature-v2")
    tools = PlatformTools(export_path=export_bundle(tmp_path / "export", features=[feature_row(), newer]),
                          model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(), "match": fixture})
    assert result.status == "NO_DATA" and "unambiguous" in result.error


def test_feature_model_request_and_fixture_match_ids_must_agree(tmp_path):
    catalog, _ = elo_catalog(tmp_path)
    tools = PlatformTools(export_path=export_bundle(tmp_path / "export", features=[feature_row()]),
                          model_id="constructed-elo", model_catalog=catalog)
    fixture = tools.execute("query_matches", query_inputs()).data[0]
    result = tools.execute("get_features", {**query_inputs(match_id="constructed:wrong"), "match": fixture})
    assert result.status != "OK"


def test_reconstruction_retains_late_observation_and_runs_under_explicit_assumptions(tmp_path):
    catalog, service = elo_catalog(tmp_path, "historical_reconstruction")
    snapshot = change_manifest(feature_row(information_mode="historical_reconstruction"), information_mode="historical_reconstruction")
    path = export_bundle(tmp_path / "export", features=[snapshot], mode="historical_reconstruction", observed="2026-09-21T12:00:00Z")
    tools = PlatformTools(export_path=path, model_id="constructed-elo", model_catalog=catalog,
                          information_mode="historical_reconstruction")
    evidence = tools.execute("read_evidence", query_inputs()).data[0]
    assert evidence["observed_at"] == "2026-09-21T12:00:00Z"
    assert evidence["reconstruction"]["basis"]["result_delay_hours"] == 3
    workflow = Workflow(tools, store_dir=tmp_path / "runs")
    run = workflow.start(MATCH, CUTOFF, Budget(max_queries=0))
    result = workflow.run_to_completion(run["run_id"])
    assert result["state"] == "STOPPED", result["reason"]
    assert result["prediction"]["information_mode"] == "historical_reconstruction"
