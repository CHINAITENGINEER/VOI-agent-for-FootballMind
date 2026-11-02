"""The fixed EPL source preset must never silently query another league's pages."""
from types import SimpleNamespace

import pytest

from football_agent import learned_voi_cli


@pytest.mark.parametrize("leagues", [["La_liga"], [], [None], ["EPL", "La_liga"]])
def test_wrong_or_ambiguous_source_league_rejected_before_loading_fixture(monkeypatch, leagues):
    manifest = {"source": "understat", "season": "2026",
                "source_evidence": [{"original_context": {"league": league}} for league in leagues]}
    monkeypatch.setattr(learned_voi_cli, "verify_completed_export",
                        lambda _: SimpleNamespace(manifest=manifest))

    def should_not_load(_):
        pytest.fail("An unsupported source preset reached fixture construction")

    monkeypatch.setattr(learned_voi_cli, "load_completed_history_export", should_not_load)
    with pytest.raises(ValueError, match="unambiguous EPL source lineage"):
        learned_voi_cli.prepare_fixture("synthetic-unused-path", now="2026-10-06T09:00:00Z")
