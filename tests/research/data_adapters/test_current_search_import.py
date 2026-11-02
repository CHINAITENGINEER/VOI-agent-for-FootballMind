from datetime import datetime, timezone
from football_spider.storage.raw_archive import RawArchive, json_bytes
from football_research.data_adapters.current_search_import import import_search_observation
from football_research.data_adapters.phase3_historical_panel import load_rows


def test_shared_search_return_keeps_order_unknown_cost_and_fixture_namespaces(tmp_path):
    archive = RawArchive(tmp_path)
    payload = {"content": [{"type": "text", "text": "First (https://club.test/a)\n【ref0】 [wordlim: 200]\nSnippet A\n-----\nOther match (https://club.test/b)\n【ref1】 [wordlim: 200]\nUnrelated snippet"}]}
    obs = archive.save("current_web_search_tool", "https://tool.test/search", {"query_batch": ["A", "B"]},
        json_bytes(payload), observed_at=datetime.now(timezone.utc),
        context={"requested_at": None, "request_clock_status": "not_measured"})
    assert import_search_observation(tmp_path, obs, archive, ["m1", "m2"]) == 2
    import_search_observation(tmp_path, obs, archive, ["m1", "m2"])
    menus = load_rows(tmp_path, "discovery.jsonl")
    assert len(menus) == 2
    assert [c["title"] for c in menus[0]["visible_candidates"]] == ["First", "Other match"]
    assert len({c["candidate_id"] for m in menus for c in m["visible_candidates"]}) == 4
    assert all(m["original_agent_menu_proven"] is False for m in menus)
    attempts = load_rows(tmp_path, "attempts.jsonl")
    assert len(attempts) == 1
    assert attempts[0]["request_count"] is None
    assert attempts[0]["tool_call_count"] == 1
    assert attempts[0]["requested_at"] is None
