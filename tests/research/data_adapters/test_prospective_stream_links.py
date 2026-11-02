from __future__ import annotations

from copy import deepcopy
import json

from football_research.data_adapters.prospective_stream import (
    _document_link_metadata,
    _visible_document_links,
    export_batch,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_document_link_metadata_retains_unknowns_and_existing_values_exactly():
    document = {"document_id": "d", "text": "body", "links": [], "filter_status": None,
                "extraction_version": 3, "custom_filter_counts": {"retained": 0, "provider": None}}
    saved = deepcopy(document)
    projected = _document_link_metadata(document)
    assert projected["links"] == []
    assert projected["filter_status"] is None
    assert projected["extraction_version"] == 3
    assert "links_truncated" not in projected
    assert "eligible_link_count" not in projected
    assert projected["runtime_document_metadata"]["custom_filter_counts"] == document["custom_filter_counts"]
    assert document == saved


def test_visible_link_projection_distinguishes_missing_empty_and_partial_context():
    assert _visible_document_links({})["visible_document_links"] is None
    empty = _visible_document_links({"public_before": {"documents": []}})
    assert empty["visible_document_links"] == []
    assert empty["visible_document_links_status"] == "recorded"
    partial = _visible_document_links({"public_before": {"documents": [{"document_id": "missing-links"}]}})
    assert partial["visible_document_links_status"] == "partially_recorded"
    assert partial["visible_document_links"] == []


def test_export_links_use_only_actual_predecision_context_and_do_not_add_actions(tmp_path):
    batch = tmp_path / "batch"
    folder = batch / "match-1"
    config = {"information_mode": "historical_reconstruction", "created_at": "2026-10-09T09:00:00Z",
        "initial_state": {"match_id": "m1", "base": {}},
        "evidence_projection": {"match": {"match_id": "m1", "home_team": "Home", "away_team": "Away",
            "competition": "Test", "season": "unknown", "kickoff_at": "2026-10-10T12:00:00Z",
            "fixture_available_at": "2026-10-09T08:00:00Z", "prediction_cutoff": "2026-10-10T11:59:00Z"}}}
    old = {"document_id": "old", "links": [{"url": "https://allowed.example/old", "title": "Old"}]}
    response_links = [{"url": "https://allowed.example/new-b", "title": "Second", "context": {"anchor_index": 8}},
                      {"url": "https://allowed.example/new-a", "title": "First"}]
    returned = {"document_id": "new", "url": "https://allowed.example/news", "text": "source text",
        "kind": "page", "status": "OK", "links": response_links, "links_truncated": True,
        "eligible_link_count": 121, "text_truncated": False, "extraction_version": 3,
        "available_at": "2026-10-09T09:01:02Z", "observed_at": "2026-10-09T09:01:01Z"}
    empty = {"document_id": "search", "url": "https://search.example/rss", "text": "", "kind": "discovery",
        "status": "NO_DATA", "links": [], "filter_status": "all_disallowed_hosts", "text_truncated": False,
        "available_at": "2026-10-09T09:02:02Z", "observed_at": "2026-10-09T09:02:01Z"}
    action = {"action_id": "read", "tool": "read_url", "url": returned["url"]}
    cycles = []
    for step, documents, result in [(0, [old], returned), (1, [old, returned], empty)]:
        cycles.append({"decision_at": f"2026-10-09T09:0{step + 1}:00Z", "generated_actions": [action],
            "selected_action": action, "public_before": {"documents": documents}, "public_after": {},
            "execution_status": "returned", "actual_cost": {"requests": 1}})
        _save(folder / f"receipt-{step:03d}.json", {"status": "OK", "document": result,
            "receipt": {"observation": {"observed_at": result["observed_at"], "context": {}}}})
    _save(folder / "configuration.json", config)
    _save(folder / "run.json", {"cycles": cycles, "terminal_selection": {"reason": "budget"}})
    output = tmp_path / "output"
    report = export_batch(batch, output)
    discoveries = _rows(output / "discovery.jsonl")
    assert len(discoveries) == 2
    first = discoveries[0]["visible_document_links"]
    assert [(link["document_id"], link["url"]) for link in first] == [("old", old["links"][0]["url"])]
    second = discoveries[1]["visible_document_links"]
    assert [(link["document_order"], link["link_order"]) for link in second] == [(0, 0), (1, 0), (1, 1)]
    assert [link["link"] for link in second[1:]] == response_links
    assert all(len(row["visible_candidates"]) == 1 for row in discoveries)
    assert len(_rows(output / "attempts.jsonl")) == 2
    documents = {row["document_id"]: row for row in _rows(output / "documents.jsonl")}
    assert documents["new"]["links"] == response_links
    assert documents["new"]["links_truncated"] is True
    assert documents["new"]["eligible_link_count"] == 121
    assert documents["new"]["extraction_version"] == 3
    assert documents["search"]["status"] == "NO_DATA"
    assert documents["search"]["filter_status"] == "all_disallowed_hosts"
    assert "links_truncated" not in documents["search"]
    assert report["discovery_document_statuses"] == {"NO_DATA": 1}
    assert "does not prove the upstream provider returned zero items" in report["retrieval_no_data_boundary"]
    assert report["documents_with_link_truncation_unknown"] == 1
    assert report["documents_with_links_truncated"] == 1
