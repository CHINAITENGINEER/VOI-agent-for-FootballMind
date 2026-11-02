"""Archived failure regression + real frozen f, mocked transport, zero HTTP."""
from copy import deepcopy
import json

import pytest
from jsonschema import ValidationError

from football_agent.semantic_request_binding import (
    VERSION, provider_input, bind_response, filter_language_controls,
)
from football_research.acquisition.semantic_f import public_input, validate_response
from football_research.acquisition.news_link_projection import project_news_links
from tests.research.acquisition.test_entity_semantic_f_collection import (
    ROOT, TRACE_FILE, QUOTE, no_network, collect_entity, replay_without_provider,
    exported_quality_rows,
)
from tests.research.acquisition.test_verified_collection import verified_inputs


def source_input():
    return public_input({"physical_match_id": "test-only:one", "home_name": "Spain",
        "away_name": "Croatia", "kickoff": "2026-10-11T13:00:00Z"},
        {"url": "https://source.example.test/news/one", "title": "Team news", "text": QUOTE}, "body")


def assertions():
    return [{"team": "home", "entity": "Alex Test", "kind": "availability",
        "value": "unavailable", "scope": "this_match", "statement": QUOTE, "quotes": [QUOTE]}]


def test_host_binding_keeps_validated_quotes_without_asking_model_to_echo_id():
    public = source_input()
    request = provider_input(public)
    assert "input_id" not in request
    wire = {"schema": VERSION, "claims": assertions()}
    bound, binding = bind_response(public, request, wire)
    assert bound["input_id"] == public["input_id"]
    assert validate_response(public, bound) == assertions()
    assert wire == {"schema": VERSION, "claims": assertions()}
    assert binding["binding_basis"] == "synchronous_call_return_not_model_echo"


@pytest.mark.parametrize("corruption", ["legacy_hash", "wrong_schema", "request_swap", "bad_quote", "bad_kind"])
def test_binding_does_not_relax_source_validation(corruption):
    public, wire = source_input(), {"schema": VERSION, "claims": assertions()}
    request = provider_input(public)
    if corruption == "legacy_hash":
        wire["input_id"] = public["input_id"]
    elif corruption == "wrong_schema":
        wire["schema"] = public["schema"]
    elif corruption == "request_swap":
        request["text"] = "Different acquired document"
    elif corruption == "bad_quote":
        wire["claims"][0]["quotes"] = ["Unacquired football assertion"]
    else:
        wire["claims"][0]["value"] = "must_win"
    with pytest.raises((ValueError, ValidationError)):
        bind_response(public, request, wire)


def test_actual_archived_failure_stays_failure_and_four_navigation_labels_are_excluded():
    path = ROOT / "var/f-entity-semantic-g-shadow-cumulative-b-20261011-slot01-v1/batch/match-3003903" / TRACE_FILE
    if not path.is_file():
        pytest.skip("Independent failed run is not included in this checkout")
    raw = path.read_bytes()
    trace = json.loads(raw)
    attempts = [a for e in trace["events"] for a in e["attempts"]]
    assert [a["input"]["title"] for a in attempts] == ["English", "ع", "Español", "Thai"]
    failed = attempts[-1]
    with pytest.raises(ValueError, match="Source assertion input binding differs"):
        validate_response(failed["input"], failed["response"])
    # The new adapter may not pretend an old response was a new request.
    with pytest.raises(ValidationError):
        bind_response(failed["input"], provider_input(failed["input"]), failed["response"])
    links = [{"url": a["input"]["url"], "title": a["input"]["title"]} for a in attempts]
    old_kept, _ = project_news_links(links)
    assert len(old_kept) == 4
    kept, audit = filter_language_controls(old_kept)
    assert kept == [] and audit["excluded_labels"] == 4
    assert path.read_bytes() == raw


def test_navigation_filter_preserves_real_unrelated_titles_and_input_bytes():
    links = [{"url": "https://example.test/news/x", "title": title} for title in
             ["English", "  Español  ", "English striker wins award", "Other club match preview", "Injury update"]]
    old = deepcopy(links)
    kept, audit = filter_language_controls(links)
    assert links == old
    assert [x["title"] for x in kept] == [x["title"] for x in links[2:]]
    assert audit["excluded_labels"] == 2


@pytest.mark.parametrize("mode", ["ok", "bad_quote", "failure", "late_provider"])
def test_v2_real_f_collection_replay_and_quality_targets(verified_inputs, monkeypatch, mode):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch,
        mode=mode, protocol_version="v2", language_links=True)
    assert calls and all("input_id" not in c for c in calls)
    assert any(c["channel"] == "body" for c in calls)
    assert not any(c["text"] in {"English", "ع", "Español", "Thai"} for c in calls)
    trace = json.loads((folder / TRACE_FILE).read_bytes())
    assert trace["schema"] == "entity-semantic-shadow-extraction-trace-v2"
    assert trace["policy_version"].startswith("entity-semantic-market-shadow-f-only-policy-v2:")
    good = [a for e in trace["events"] for a in e["attempts"] if a["status"] == "OK"]
    if mode in {"ok", "late_provider"}:
        assert good
    for attempt in good:
        bound, binding = bind_response(attempt["input"], attempt["provider_input"], attempt["provider_response"])
        assert bound == attempt["response"] and binding == attempt["response_binding"]
    replay_without_provider(folder, verified_inputs, monkeypatch)
    derived = exported_quality_rows(folder, verified_inputs)
    eligible = [r for r in derived if r.get("g_quality_target", {}).get("status") == "eligible"]
    if mode == "ok":
        assert len(eligible) == 1
        assert eligible[0]["g_quality_target"]["realized_signed_gain"] != 0
    else:
        assert eligible == []
