"""The historical delivery must work without a live deployment or user topics."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from football_agent import footballmind as cli


def test_historical_catalog_does_not_load_live_deployment(monkeypatch, capsys):
    rows = [{"match_id": "m1", "home": "Home", "away": "Away"}, {"match_id": "m2"}]
    monkeypatch.setitem(sys.modules, "football_agent.historical_analysis",
        SimpleNamespace(load_catalog=lambda **kw: rows, analyze_match=None))
    monkeypatch.setattr(cli, "load_deployment", lambda *a, **k: pytest.fail("historical depends on live deployment"))
    assert cli.main(["historical-matches", "--limit", "1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total"] == 2
    assert result["matches"] == rows[:1]
    assert result["network_requests"] == 0


def test_match_only_analysis_uses_frozen_defaults_and_automatic_output(monkeypatch, tmp_path, capsys):
    calls = []

    def analyze(match, **kwargs):
        calls.append((match, kwargs))
        return {"mode": "historical_reconstruction", "match_id": "m1"}

    monkeypatch.setitem(sys.modules, "football_agent.historical_analysis",
        SimpleNamespace(load_catalog=None, analyze_match=analyze))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["historical-analyze", "--match", "Home vs Away"]) == 0
    match, kwargs = calls[0]
    assert match == "Home vs Away"
    assert kwargs["max_queries"] is None and kwargs["lambda_money"] is None
    assert Path(kwargs["output"]).parent == tmp_path / "var/submission/runs"
    assert "report.md" in capsys.readouterr().out


def test_catalog_rejects_invalid_limit_before_loading(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "football_agent.historical_analysis",
        SimpleNamespace(load_catalog=lambda **kw: pytest.fail("invalid limit reached loading"), analyze_match=None))
    assert cli.main(["historical-matches", "--limit", "0"]) == 2
    assert "limit" in capsys.readouterr().err


def test_open_archive_entry_keeps_its_own_identity_and_default_output(monkeypatch, tmp_path, capsys):
    calls = []
    rows = [{"match_id": "official-m1", "home": "Actual Home", "away": "Actual Away"}]

    def replay(match, **kwargs):
        calls.append((match, kwargs))
        return {"mode": "archived_open_information", "match_id": "official-m1"}

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "football_agent.open_archive_analysis",
        SimpleNamespace(load_open_catalog=lambda **kw: rows, replay_open_match=replay))
    assert cli.main(["open-matches", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["matches"] == rows
    assert cli.main(["open-replay", "--match", "official-m1"]) == 0
    assert calls[0][0] == "official-m1"
    assert calls[0][1]["output"].parent == tmp_path / "var/submission/open-runs"
