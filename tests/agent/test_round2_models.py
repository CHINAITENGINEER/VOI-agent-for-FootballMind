"""Round-two model routing and market grouping, with explicit fixture evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from football_agent.adapters import LocalArchiveTools, digest
from football_agent.model_catalog import ModelCatalog
from football_research.baselines.api import PredictionService


CUTOFF = "2026-09-20T11:25:00Z"


def test_platform_qualified_imports_need_no_agent_path_bridge():
    code = """
import sys
from football_agent.adapters import _platform
before = list(sys.path)
RawArchive, normalize, Observation = _platform()
assert sys.path == before
assert RawArchive.__module__ == 'football_spider.storage.raw_archive'
assert normalize.__module__ == 'football_spider.normalization.sources'
assert Observation.__module__ == 'football_spider.models.contracts'
assert 'models.contracts' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def fixture():
    return {"match_id": "constructed:unseen", "competition_id": "spftll:competition:36",
            "home_team_id": "spftll:team:old-home", "away_team_id": "spftll:team:old-away",
            "kickoff": "2026-09-20T13:00:00Z", "fixture_available_at": "2026-09-20T11:00:00Z",
            "information_mode": "operational", "score_period": "regulation", "status": "scheduled",
            "input_manifest": ["constructed-fixture-only-test"]}


def args():
    match = fixture()
    return {"match_id": match["match_id"], "match": match, "prediction_cutoff": CUTOFF, "evidence": []}


def market_evidence(provider="provider-a", bookmaker="book-a", prices=(2., 3., 4.),
                    available="2026-09-20T11:00:00Z", source_time=None, key="quote-a"):
    dimensions = {"provider": provider, "bookmaker": bookmaker, "market_type": "1x2",
                  "period": "regulation", "line": None,
                  "settlement_rule": "regulation_including_stoppage", "source_timestamp": source_time}
    content = {**dimensions, "quotes": [{**dimensions, "selection": selection, "decimal_price": price}
                                        for selection, price in zip(("home", "draw", "away"), prices)]}
    return {"evidence_id": key, "version": digest(content), "match_id": fixture()["match_id"],
            "source": provider, "kind": "odds", "observed_at": available, "available_at": available,
            "information_mode": "operational", "content": content, "raw_ref": "constructed:" + key}


@pytest.mark.parametrize("model_id", ["frequency-england", "poisson-england", "dixon-coles-england"])
def test_real_trained_models_predict_without_odds_and_equal_service(model_id):
    tools = LocalArchiveTools(model_id=model_id)
    inputs = args()
    envelope = tools.execute("get_features", inputs)
    assert envelope.status == "OK", envelope.error
    assert envelope.data["feature_snapshot_required"] is False
    assert envelope.data["missing"] == {}
    assert not any(key.startswith("odds_") for key in envelope.data)
    result = tools.execute("predict", {**inputs, "features": envelope.data})
    direct = PredictionService.load(tools.model_artifact).predict(inputs["match"], prediction_cutoff=CUTOFF)
    assert result.status == "OK", result.error
    assert result.data == direct.model_dump(mode="json")
    assert tools.execute("predict", inputs).data == result.data


def test_market_requires_quotes_but_trained_model_capability_does_not():
    market = LocalArchiveTools(model_id="market")
    frequency = LocalArchiveTools(model_id="frequency-england")
    assert market.execute("get_features", args()).status == "NO_DATA"
    assert market.capabilities()["required_inputs"]["odds_required"] is True
    assert frequency.capabilities()["required_inputs"]["odds_required"] is False
    assert frequency.capabilities()["trained_at"] == "2026-09-20T11:22:02+00:00"


@pytest.mark.parametrize("model_id", ["../models/anything", "D:/file.json", "market.json", "unknown", None])
def test_model_ids_are_allowlist_keys_never_paths(model_id):
    with pytest.raises(ValueError, match="Unknown model ID"):
        ModelCatalog().resolve(model_id)


def test_catalog_rejects_unpinned_registry_artifact(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"models": [{"model_id": "bad", "model_name": "frequency",
                                             "model_artifact": "arbitrary.json"}]}))
    with pytest.raises(ValueError, match="pinned SHA256"):
        ModelCatalog(path)


def test_catalog_checks_artifact_hash_on_every_resolve(tmp_path):
    original = ModelCatalog().resolve("frequency-england")
    artifact = tmp_path / "model.json"
    artifact.write_bytes(__import__("pathlib").Path(original["model_artifact"]).read_bytes())
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"models": [{**original, "model_id": "local", "model_artifact": "model.json"}]}))
    catalog = ModelCatalog(registry)
    assert catalog.resolve("local")["artifact_sha256"] == original["artifact_sha256"]
    data = json.loads(artifact.read_text())
    data["model_state"]["global_counts"][0] += 1
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        catalog.resolve("local")


@pytest.mark.parametrize("field,value,reason", [
    ("match_id", "wrong:id", "another match"),
    ("competition_id", "wrong:competition", "different competition"),
    ("information_mode", "historical_reconstruction", "information mode"),
    ("fixture_available_at", "2026-09-20T11:26:00Z", "unavailable"),
])
def test_trained_model_checks_fixture_identity_and_time(field, value, reason):
    inputs = args()
    inputs["match"][field] = value
    result = LocalArchiveTools(model_id="frequency-england").execute("predict", inputs)
    assert result.status == "NO_DATA"
    assert reason in result.error


def test_training_after_cutoff_stays_no_data():
    inputs = {**args(), "prediction_cutoff": "2026-09-20T11:21:00Z"}
    result = LocalArchiveTools(model_id="frequency-england").execute("predict", inputs)
    assert result.status == "NO_DATA"
    assert "trained after" in result.error


def test_independent_bookmaker_disagreement_is_not_conflict():
    inputs = {**args(), "evidence": [market_evidence(bookmaker="z-book", prices=(1.5, 3., 5.), key="z"),
                                    market_evidence(bookmaker="a-book", prices=(2., 3., 4.), key="a")]}
    result = LocalArchiveTools().execute("get_features", inputs)
    assert result.status == "OK", result.error
    assert result.data["bookmaker"] == "a-book"
    assert result.data["independent_market_groups"] == 2
    assert [result.data[k] for k in ("odds_home", "odds_draw", "odds_away")] == [2., 3., 4.]


def test_latest_complete_quote_precedes_stable_publisher_priority():
    inputs = {**args(), "evidence": [market_evidence(bookmaker="a-book"),
                  market_evidence(bookmaker="z-book", prices=(1.5, 3., 5.), available="2026-09-20T11:20:00Z", key="z")]}
    result = LocalArchiveTools().execute("get_features", inputs)
    assert result.status == "OK"
    assert result.data["bookmaker"] == "z-book"


def test_equivalent_timestamp_serializations_and_latest_quote_version():
    old = market_evidence(source_time="2026-09-20T10:59:00Z", key="a")
    new = market_evidence(source_time="2026-09-20T11:00:00+00:00", prices=(1.8, 3.4, 4.), key="z")
    for quote in new["content"]["quotes"]:
        quote["source_timestamp"] = "2026-09-20T11:00:00Z"
    result = LocalArchiveTools().execute("get_features", {**args(), "evidence": [old, new]})
    assert result.status == "OK", result.error
    assert result.data["odds_snapshot_id"] == "z"


def test_same_market_fact_version_disagreement_is_quarantined():
    inputs = {**args(), "evidence": [market_evidence(), market_evidence(prices=(1.5, 3., 5.), key="b")]}
    result = LocalArchiveTools().execute("get_features", inputs)
    assert result.status == "FAILED"
    assert result.error["evidence_ids"] == [["quote-a", "b"]]


@pytest.mark.parametrize("field,value", [("provider", "other"), ("bookmaker", "other"),
                                        ("period", "first_half"), ("line", 1),
                                        ("source_timestamp", "2026-09-20T10:59:00Z"),
                                        ("settlement_rule", "includes_extra_time")])
def test_market_does_not_stitch_quote_components(field, value):
    evidence = market_evidence()
    evidence["content"]["quotes"][0][field] = value
    result = LocalArchiveTools().execute("get_features", {**args(), "evidence": [evidence]})
    assert result.status == "NO_DATA"
    assert result.error["rejected_inconsistent_groups"] == ["quote-a"]


def test_feature_envelope_hash_and_selected_model_are_verified():
    tools = LocalArchiveTools(model_id="frequency-england")
    envelope = tools.execute("get_features", args()).data
    tampered = {**envelope, "input_manifest": ["changed"]}
    result = tools.execute("predict", {**args(), "features": tampered})
    assert result.status == "NO_DATA" and "hash mismatch" in result.error
    result = tools.execute("predict", {**args(), "features": {**envelope, "model_id": "poisson-england"}})
    assert result.status == "NO_DATA" and "another selected model" in result.error
