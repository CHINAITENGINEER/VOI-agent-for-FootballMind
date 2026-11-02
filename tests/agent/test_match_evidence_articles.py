"""Versioned already-visible article evidence, without inferred football facts."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.match_evidence import build_match_evidence, ARTICLES_VERSION, VERSION, LEGACY_VERSION, _build_match_evidence
from test_match_evidence import MATCH, AS_OF, document


def article(parts, *, docid="article-1", role="article", **updates):
    segments, offset = [], 0
    for part in parts:
        content = {"text": part} if isinstance(part, str) else deepcopy(part)
        start = offset+(1 if segments else 0)
        content = {"source_published_at": "2026-10-06T07:00:00Z", "source_modified_at": None,
                   "content_source": "json_ld_article_body", "json_pointer": f"script[0].liveBlogUpdate[{len(segments)}].articleBody",
                   **content, "start": start, "end": start+len(content["text"])}
        segments.append(content)
        offset = content["end"]
    text = "\n".join(s["text"] for s in segments)
    return document(text, docid, kind="article", structured_news={
        "version": "structured-news-discovery-v2", "role": role, "segments": segments,
        "source_language": "en", "language_source": "html@lang"}, **updates)


def project(documents, **kwargs):
    return build_match_evidence(MATCH, documents, as_of=kwargs.pop("as_of", AS_OF), version=ARTICLES_VERSION, **kwargs)


def test_real_article_kind_reaches_existing_followup_contract():
    source = article(["Arsenal v Leeds on 2026-10-10: Arsenal may change the lineup because Kai Havertz has an injury."])
    original = deepcopy(source)
    old = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert not old["evidence"]
    report = project([source])
    assert source == original
    assert report["schema_version"] == ARTICLES_VERSION
    assert report["evidence"] and report["state"]["followup_candidates"]
    candidate = report["state"]["followup_candidates"][0]
    assert "Havertz" in candidate["query"] and candidate["expected_gain"] is None
    assert candidate["proposal_status"] == "grounded_query_not_value_ranked"
    for row in report["evidence"]:
        anchor = row["anchor"]
        assert source["text"][anchor["start"]:anchor["end"]] == row["statement"]
        assert row["source_published_at"] == "2026-10-06T07:00:00Z"
        assert row["normalized_predicate"] is None and row["effective_at"] is None
    assert report["semantics"]["probability_update"] is None


def test_per_liveblog_segment_times_not_replaced_with_page_latest_time():
    source = article([
        {"text": "Arsenal have reported an injury affecting the available squad.",
         "source_published_at": "2026-10-06T06:00:00Z", "source_modified_at": "2026-10-06T06:30:00Z"},
        {"text": "Leeds may announce another update about their available starting lineup.",
         "source_published_at": "2026-10-06T08:00:00Z"}], role="liveblog", source_published_at="2026-10-06T08:00:00Z")
    rows = project([source])["evidence"]
    assert [r["source_published_at"] for r in rows] == ["2026-10-06T06:00:00Z", "2026-10-06T08:00:00Z"]
    assert rows[0]["source_modified_at"] == "2026-10-06T06:30:00Z"
    assert all(r["observed_at"] == source["observed_at"] and r["available_at"] == source["available_at"] for r in rows)


@pytest.mark.parametrize("field,value", [("source_modified_at", "2026-10-06T10:00:00Z"),
    ("source_published_at", "2026-10-06T10:00:00Z"), ("source_modified_at", "2026-10-06T08:00:00")])
def test_rejected_future_or_invalid_clock_segment_does_not_leak_through_any_public_field(field, value):
    source = article(["Arsenal may update the starting lineup after further training.",
        {"text": "Arsenal SECRETSTRIKER suffered an injury and may be unavailable.", field: value}])
    report = project([source])
    assert report["evidence"]
    assert len(report["document_reviews"][0]["excluded_segments"]) == 1
    assert "SECRETSTRIKER" not in json.dumps(report)


def test_local_publication_date_does_not_invent_timezone_or_absolute_time():
    source = article([{"text": "Arsenal may update the starting lineup after further training.",
        "source_published_at": None, "source_publication_date": "2026-10-06",
        "source_publication_local_time": "12:05", "publication_raw": "06 10.2026 12:05",
        "publication_timezone": "unknown"}])
    row = project([source])["evidence"][0]
    assert row["source_published_at"] is None
    assert row["source_publication_date"] == "2026-10-06"
    assert row["source_publication_local_time"] == "12:05"
    assert row["publication_raw"] == "06 10.2026 12:05"
    assert row["publication_status"] == "segment_timezone_unknown"
    source["structured_news"]["segments"][0]["source_publication_date"] = "2026-10-08"
    assert not project([source])["evidence"]


def test_fixture_pair_and_date_cannot_be_joined_across_article_segments():
    report = project([article(["Arsenal v Leeds with the date still to be confirmed.",
                               "The Chelsea match will be played on 2026-10-10."])])
    assert report["evidence"] and report["unscoped_source_quotes"]
    assert all(r["applicability"] != "match_scoped_source_assertion" for r in report["evidence"])
    assert "fixture_confirmation" not in report["state"]["present_topics"]


def test_source_fixture_line_does_not_grant_scope_to_another_sentence():
    report = project([article(["Arsenal v Leeds on 2026-10-10. Arsenal may rotate the squad."])])
    rotation = next(r for r in report["evidence"] if "rotation" in r["topics"])
    assert rotation["applicability"] == "team_context_requires_match_binding"


def test_unclassified_foreign_body_is_visible_without_target_or_semantic_promotion():
    source = article(["Täna õhtul mängib meeste koondis ning publik saab kohtumist vaadata."])
    source["structured_news"]["source_language"] = "et"
    source["local_scope"] = {"scope": "match_specific", "semantic_truth": True}
    report = project([source])
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    row = report["unscoped_source_quotes"][0]
    assert row["statement"] == source["text"]
    assert row["applicability"] == "unscoped_source_text" and row["normalized_predicate"] is None


def test_qualification_does_not_create_rotation_weakness_or_defensive_conclusion():
    report = project([article(["Arsenal v Leeds on 2026-10-10: Arsenal have already qualified for the next round."])])
    assert "qualification" in report["state"]["present_topics"]
    assert not {"rotation", "tactics"}.intersection(report["state"]["present_topics"])
    assert all(r["expected_gain"] is None and r["normalized_predicate"] is None for r in report["evidence"])


def test_lexical_opposition_copy_and_version_edges_do_not_resolve_truth():
    yes = article(["Arsenal v Leeds on 2026-10-10: Arsenal will rotate the starting lineup."], docid="yes")
    no = article(["Arsenal v Leeds on 2026-10-10: Arsenal will not rotate the starting lineup."], docid="no", url=yes["url"],
                 observed_at="2026-10-06T10:00:00Z", available_at="2026-10-06T10:00:01Z")
    mirror = deepcopy(yes)
    mirror.update(document_id="mirror", url="https://other.example/news/copied")
    report = project([yes, no, mirror])
    assert report["state"]["conflict_signals"] and report["state"]["exact_copy_signals"]
    assert all(c["verified_contradiction"] is False and c["expected_gain"] is None for c in report["state"]["conflict_signals"])
    assert all(c["independent_corroboration"] is None for c in report["state"]["exact_copy_signals"])
    edge = report["state"]["source_version_relations"][0]
    assert edge["semantic_revision_or_supersession"] == "unknown"
    assert edge["captures"][0]["document_id"] == "yes"


@pytest.mark.parametrize("change", ["index", "search", "failed", "late", "offset", "overlap", "unknown_version", "no_segments"])
def test_non_body_unavailable_and_invalid_span_inputs_do_not_create_claims(change):
    source = article(["Arsenal may rotate the available starting lineup after training."])
    if change in {"index", "search"}: source["kind"] = change
    elif change == "failed": source["status"] = "FAILED"
    elif change == "late": source["available_at"] = "2026-10-06T13:00:00Z"
    elif change == "offset": source["structured_news"]["segments"][0]["start"] = 1
    elif change == "overlap":
        source["structured_news"]["segments"].append(deepcopy(source["structured_news"]["segments"][0]))
        result = project([source])
        assert len(result["evidence"]) == 1 and result["document_reviews"][0]["excluded_segments"]
        return
    elif change == "unknown_version": source["structured_news"]["version"] = "future-version"
    else: source["structured_news"]["segments"] = []
    report = project([source])
    assert not report["evidence"] and not report["state"]["followup_candidates"]


def test_projection_id_stable_and_bounds_explicit():
    source = article(["Arsenal may rotate their starting lineup.", "Leeds could announce another squad update."])
    first = project([source])
    later = project([source], as_of="2026-10-06T12:01:00Z")
    assert [r["evidence_id"] for r in first["evidence"]] == [r["evidence_id"] for r in later["evidence"]]
    limited = project([source], max_evidence=1)
    assert len(limited["evidence"]) == 1 and limited["state"]["extraction_truncated"]
    assert not limited["state"]["coverage_complete"]


@pytest.mark.parametrize("route,reason", [
    ("/", "index_page_for_link_discovery_only"),
    ("/en/?news=preview", "index_page_for_link_discovery_only"),
    ("/en/news", "index_page_for_link_discovery_only"),
    ("/news/page/2", "index_page_for_link_discovery_only"),
    ("/women/news/arsenal-leeds-preview", "non_target_squad_source_section"),
    ("/academy/news/arsenal-leeds-preview", "non_target_squad_source_section"),
    ("/u21/news/arsenal-leeds-preview", "non_target_squad_source_section"),
])
def test_structured_article_does_not_bypass_existing_source_section_guard(route, reason):
    source = article(["Arsenal v Leeds on 2026-10-10: Kai Havertz may miss the starting lineup with an injury."],
                     url="https://club.example"+route)
    report = project([source])
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    assert report["document_reviews"][0]["reason"] == reason


def test_explicit_womens_target_is_not_rejected_by_womens_source_route():
    identity = {**MATCH, "home_team": "Arsenal Women", "away_team": "Leeds Women"}
    source = article(["Arsenal Women v Leeds Women on 2026-10-10: Arsenal Women may rotate the starting lineup."],
                     url="https://club.example/women/news/arsenal-leeds-preview")
    report = build_match_evidence(identity, [source], as_of=AS_OF, version=ARTICLES_VERSION)
    assert report["evidence"] and report["state"]["followup_candidates"]


def test_previous_default_and_explicit_versions_remain_equal():
    docs = [document("Arsenal v Leeds on 2026-10-10: Arsenal may rotate the squad.")]
    for version in (LEGACY_VERSION, VERSION):
        expected = _build_match_evidence(MATCH, docs, as_of=AS_OF, max_evidence=64, version=version)
        assert build_match_evidence(MATCH, docs, as_of=AS_OF, version=version) == expected
    assert build_match_evidence(MATCH, docs, as_of=AS_OF) == build_match_evidence(MATCH, docs, as_of=AS_OF, version=VERSION)


def test_existing_candidate_generator_consumes_new_source_followup_only_after_read():
    from test_evidence_candidates import make, MATCH as CANDIDATE_MATCH
    source = article(["Arsenal striker Kai Havertz may miss the starting lineup with an injury."])
    generator = make()
    def public(documents):
        return {"state": {"base": {"kickoff": CANDIDATE_MATCH["kickoff_at"]}}, "documents": documents,
                "evidence": build_match_evidence(CANDIDATE_MATCH, documents, as_of=AS_OF, version=ARTICLES_VERSION)}
    before, after = generator(public([])), generator(public([source]))
    assert not any("Havertz" in a.get("query", "") for a in before)
    assert any("Havertz" in a.get("query", "") for a in after)
    assert all("expected_gain" not in action for action in after)


def test_archived_official_local_articles_keep_exact_body_spans_and_unknown_scope(monkeypatch):
    from football_agent.news_discovery import NewsDiscoveryTools
    from football_spider.models.contracts import Observation
    from football_spider.storage.raw_archive import RawArchive
    root = Path(__file__).resolve().parents[2]
    path = root/"var/parallel/agent/round184/panel-repair/panel-2981554/panel.json"
    if not path.exists():
        pytest.skip("Real article archive is not available")
    original = path.read_bytes()
    panel = json.loads(original)
    archive = RawArchive(Path(panel["archive_folder"]))
    config = panel["design"]["transport"]
    at = datetime(2026, 10, 6, 16, 36, tzinfo=timezone.utc)
    tools = NewsDiscoveryTools(archive, config["allowed_hosts"], provider=config["search_provider"],
        default_route=config["source_route_policy"]["default_route"],
        route_overrides=config["source_route_policy"]["route_overrides"],
        index_urls=config["news_discovery"]["index_urls"], projection_version=2, clock=lambda: at)
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("Unexpected network request"))
    match = panel["design"]["match_scope"]
    identity = {"match_id": match["match_id"], "home_team": "Estonia", "away_team": "Iceland",
                "kickoff_at": match["kickoff_at"], "prediction_cutoff": match["prediction_cutoff"]}
    count = 0
    for capture in panel["captures"]:
        observation = Observation.model_validate(capture["result"]["observation"])
        source = tools.project(observation, capture["prepared"])
        report = build_match_evidence(identity, [source], as_of=at, version=ARTICLES_VERSION)
        assert not report["state"]["present_topics"]
        for row in report["evidence"]+report["unscoped_source_quotes"]:
            anchor = row["anchor"]
            assert source["text"][anchor["start"]:anchor["end"]] == row["statement"]
            assert row["source_published_at"] is None
            assert row["source_publication_date"] == "2026-10-06"
            count += 1
    assert count > 10
    assert path.read_bytes() == original
