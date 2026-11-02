"""Actual captured bytes: derived documents cannot rewrite operational history."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from football_research.data_adapters.derived_text_corpus import build_corpus, verify_corpus
from football_research.data_adapters.open_text_episodes import _digest
from football_agent.learned_voi_replay import replay_saved_run
from test_verified_collection import verified_inputs, _collect

ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "var/parallel/agent/round184"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not (ROUND / "panel-repair/report.json").exists():
        pytest.skip("Real archived cohort not distributed with minimal source install")
    folder = tmp_path_factory.mktemp("derived-corpus")
    path = folder / "corpus.json"
    actors = sorted((ROUND / "prospective-panels").glob("actor-*/match-*"))
    panels = sorted((ROUND / "panel-repair").glob("panel-*/panel.json"))
    # Reprocessing after kickoff is permitted for offline text candidates;
    # neither an old actor admission nor historical processing time is invented.
    at = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
    result = build_corpus(actors, panels, output=path, clock=lambda: at)
    return path, result, actors, panels, at


def test_actual_source_reprocessing_is_offline_and_preserves_clocks_and_all_cases(corpus, monkeypatch):
    path, result, actors, panels, at = corpus
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("Offline reprocessing used network"))
    assert verify_corpus(path)["status"] == "passed"
    assert result["audit"]["frozen_panel_cases"] == 5
    assert result["audit"]["research_responses"] == result["audit"]["derived_article_candidates"] == 2
    assert result["audit"]["derived_supervised_examples"] == result["audit"]["new_operational_voi_calibration_records"] == 0
    assert len(result["inputs"]["panel_sha256"]) == 5
    for row in result["derived_records"]:
        assert row["raw_captured_before_cutoff"] and not row["derived_before_cutoff"]
        assert row["derived_available_at"] == at.isoformat()
        assert row["source_clocks"]["original_available_at"] == row["original_document"]["available_at"]
        assert row["original_qualification"]["article"] is False
        assert row["derived_document"]["source_published_at"] is None
        assert row["selection_probability"] is None and row["information_value"] is None
        assert row["risk_calibration_eligible"] is row["operational_transition_eligible"] is False
        assert row["local_scope"]["alias_status"] == "unverified_source_assertion"
        assert "history" not in row  # Sibling research reads are not an actor sequence.


def corrupt(tmp_path, original, change):
    bad = deepcopy(original)
    change(bad)
    bad["corpus_sha256"] = _digest({k: v for k, v in bad.items() if k != "corpus_sha256"})
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    return path


def test_rehashed_statistics_cannot_inflate_supervised_or_calibration_counts(corpus, tmp_path):
    path = corrupt(tmp_path, corpus[1], lambda x: x["audit"].update(derived_supervised_examples=2))
    with pytest.raises(ValueError, match="statistics"):
        verify_corpus(path)


def test_repeated_panel_cannot_inflate_case_or_response_count(corpus, tmp_path):
    def change(x):
        x["inputs"]["panel_paths"].append(x["inputs"]["panel_paths"][-1])
        x["derived_records"].extend(deepcopy(x["derived_records"]))
    path = corrupt(tmp_path, corpus[1], change)
    with pytest.raises(ValueError, match="unique"):
        verify_corpus(path)


def test_empty_panels_are_hash_bound_too(corpus, tmp_path):
    def change(x):
        key = next(p for p in x["inputs"]["panel_paths"] if "2981455" in p)
        x["inputs"]["panel_sha256"][key] = "0" * 64
    path = corrupt(tmp_path, corpus[1], change)
    with pytest.raises(ValueError, match="Frozen original panel"):
        verify_corpus(path)


def test_declared_result_cannot_replace_independent_score(corpus, tmp_path):
    def change(x):
        x["supervision_sidecar"][0].update(supervised_text_eligible=True, label={"outcome": 0}, exclusion_reasons=[])
    path = corrupt(tmp_path, corpus[1], change)
    with pytest.raises(ValueError, match="supervision"):
        verify_corpus(path)


def test_new_content_cannot_be_claimed_available_before_original_panel(corpus, tmp_path):
    path = corrupt(tmp_path, corpus[1], lambda x: x["derived_records"][0].update(
        derived_available_at=x["derived_records"][0]["source_clocks"]["original_available_at"]))
    with pytest.raises(ValueError, match="predate"):
        verify_corpus(path)


def test_corpus_cannot_precede_new_projection(corpus, tmp_path):
    path = corrupt(tmp_path, corpus[1], lambda x: x.update(built_at="2026-10-06T15:00:00+00:00"))
    with pytest.raises(ValueError, match="predates"):
        verify_corpus(path)


def test_build_never_overwrites_existing_corpus(corpus):
    path, _, actors, panels, at = corpus
    with pytest.raises(FileExistsError):
        build_corpus(actors, panels, output=path, clock=lambda: at)


def test_prospective_v2_news_projection_is_opt_in_and_replays(verified_inputs, monkeypatch):
    folder, calls, report = _collect(verified_inputs, monkeypatch, news_projection_version=2)
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    assert config["supervisor_metadata"]["transport"]["news_discovery"]["version"] == "structured-news-discovery-v2"
    assert json.loads((folder.parent / "batch-configuration.json").read_text(encoding="utf-8"))["news_projection_version"] == 2
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert any(d["kind"] == "article" for d in run["documents"])
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("Replay used network"))
    report = replay_saved_run(folder, output=folder.parent / "new-version-replay", project_root=ROOT)
    assert report["identical_replay"] and report["replay_network_requests"] == 0
