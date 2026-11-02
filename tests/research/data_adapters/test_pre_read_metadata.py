from copy import deepcopy
from football_research.data_adapters.pre_read_metadata import candidate_pre_read_metadata
from football_platform.information.stream_validator import _pre_read_metadata


def test_metadata_uses_only_predecision_labels_not_returned_body():
    action = {"action_id": "a", "url": "https://club.test/story"}
    cycle = {"decision_at": "2026-10-09T11:00:00Z", "public_before": {"documents": [{
        "document_id": "directory", "raw_sha256": "a" * 64, "available_at": "2026-10-09T10:59:00Z",
        "links": [{"url": action["url"], "title": "Title before read", "snippet": "Already seen fact",
                   "observed_anchors": [{"text": "Literal anchor", "anchor_index": 2}]}]}]},
        "public_state_after": {"documents": [{"title": "Changed later", "text": "Never backfill"}]}}
    original = deepcopy(cycle)
    row = candidate_pre_read_metadata(cycle, action)
    assert [x["text"] for x in row["observed_labels"]] == ["Title before read", "Already seen fact", "Literal anchor"]
    assert row["pre_read_metadata"]["target_body_used"] is False
    assert row["pre_read_metadata"]["f_consumed"] is None
    assert row["pre_read_metadata"]["observations"][0]["parent"]["raw_sha256"] == "a" * 64
    assert cycle == original
    issues = []
    _pre_read_metadata(row, {"observed_at": cycle["decision_at"]}, "candidate", issues)
    assert issues == []
    row["pre_read_metadata"]["target_body_used"] = True
    _pre_read_metadata(row, {"observed_at": cycle["decision_at"]}, "candidate", issues)
    assert any(x.code == "pre_read_body_leakage" for x in issues)


def test_context_only_anchor_and_unknown_are_preserved_without_inventing_title():
    cycle = {"public_before": {"candidate_contexts": {"by_action_id": {"a": {"observations": [
        {"parent": {"document_id": "p"}, "link_index": 4, "observed_anchors": [{"text": "Observed"}]}]}}}}}
    known = candidate_pre_read_metadata(cycle, {"action_id": "a"})
    missing = candidate_pre_read_metadata(cycle, {"action_id": "b", "title": "not a recorded link"})
    assert known["observed_labels"][0]["text"] == "Observed"
    assert missing["observed_labels"] == []
    assert missing["pre_read_metadata"]["status"] == "not_recorded"
