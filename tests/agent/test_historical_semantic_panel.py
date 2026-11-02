"""Historical panel provenance checks; no target result is allowed into actor state."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "var/submission/p7-historical-semantic/case-selection/panel.json"
SELECTION = ROOT / "var/submission/p7-historical-semantic/case-selection/selection.json"


def test_historical_panel_excludes_target_label_and_preserves_source_segments():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert panel["information_mode"] == "historical_reconstruction"
    assert panel["case_status"] == "mechanism_case_not_independent_effectiveness_evaluation"
    assert len(panel["documents"]) >= 2
    forbidden = {"outcome", "label_provenance", "home_goals", "away_goals", "result_available_at", "prior_probabilities", "VoI"}
    assert forbidden.isdisjoint(panel["actor_visibility_contract"]["allowed"])
    assert forbidden.issubset(set(panel["actor_visibility_contract"]["excluded"]))
    assert selection["audit"]["selection_used_outcome_or_label"] is False
    for doc in panel["documents"]:
        assert doc.get("source_published_at") < panel["fixture"]["kickoff"]
        assert doc["available_at"] is None
        assert doc["structured_news"]["segments"]
        assert all(seg["text"] in doc["text"] for seg in doc["structured_news"]["segments"])
        raw = ROOT / doc["raw_path"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == doc["raw_sha256"]


def test_historical_selection_is_metadata_rule_and_case_is_fixed():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    rule = selection["selection_rule"]
    assert rule["version"] == "historical-case-selection-rule-v1"
    assert selection["selected_source_index"] == 20
    assert selection["audit"]["actual_acquisition_clock_backfilled"] is False
