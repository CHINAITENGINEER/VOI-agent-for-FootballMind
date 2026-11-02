"""Real archived requests exercise discovery, UNKNOWN and end-of-run reporting."""
from hashlib import sha256
import json
import socket

import pytest

from football_agent.open_archive_analysis import ROOT, load_open_catalog, replay_open_match


def _no_network(*args, **kwargs):
    raise AssertionError("Open archive experience must never access the network")


def test_catalog_is_six_real_fixture_identities_without_results():
    catalog = load_open_catalog()
    assert len(catalog) == 6
    assert len({item["match_id"] for item in catalog}) == 6
    assert all(item["mode"] == "real_open_archive_replay" for item in catalog)
    assert all("constructed" not in item["run_directory"] for item in catalog)
    assert all(not ({"outcome", "home_goals", "away_goals", "label"} & item.keys()) for item in catalog)


@pytest.mark.parametrize("match", ["SJK vs FC Lahti", "veikkausliiga:2026:match:4353570"])
def test_real_open_returns_recompute_candidates_and_keep_unknown(match, tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    output = tmp_path / "experience"
    analysis = replay_open_match(match, output=output)
    assert analysis["status"] == "completed"
    assert analysis["summary"]["archived_return_count"] == 4
    assert analysis["summary"]["newly_discovered_count"] >= 12
    assert analysis["summary"]["new_source_quote_count"] > 0
    assert analysis["summary"]["unknown_exploration_selections"] == 4
    assert analysis["summary"]["computed_selections"] == 0
    assert analysis["summary"]["replay_network_requests"] == 0
    assert analysis["initial_probabilities"] is None
    assert analysis["final_probabilities"] is None
    assert not analysis["historical_two_tool_model_attached"]
    assert analysis["terminal_selection"]["stop_certificate"] is None
    assert all(s["estimated_net_voi"] is None for s in analysis["steps"])
    assert "本次分析已结束" in (output / "report.md").read_text(encoding="utf-8")
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8")) == analysis
    original = ROOT / analysis["match"]["run_directory"] / "run.json"
    replay = output / "replay/run.json"
    assert sha256(original.read_bytes()).digest() == sha256(replay.read_bytes()).digest()
    with pytest.raises(FileExistsError):
        replay_open_match(match, output=output)


@pytest.mark.parametrize("match", ["", "Arsenal vs Leeds", "SJK", "unknown-id"])
def test_unknown_fixture_is_not_silently_replaced(match, tmp_path):
    output = tmp_path / "unmatched"
    with pytest.raises(ValueError):
        replay_open_match(match, output=output)
    assert not output.exists()
