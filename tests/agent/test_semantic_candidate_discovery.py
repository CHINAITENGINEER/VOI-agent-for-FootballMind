"""Nomination capability boundaries; not semantic relevance accuracy claims."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from football_agent.semantic_candidate_discovery import (
    CATALOG_LIMIT, CatalogDiscoveryProtocol, discovery_catalog,
    validate_catalog_discovery,
)


def article(index=0, **extra):
    return {"action_id": f"read-{index}", "tool": "read_url",
            "url": f"https://official.test/news/article-{index}",
            "title": f"Observed title {index}", "parent_url": "https://official.test/news",
            "available_at": "2026-10-09T08:00:00Z", **extra}


def nomination(*ids, queries=None):
    return {"schema": CatalogDiscoveryProtocol.version, "queries": queries or [],
            "article_candidates": [{"action_id": action_id,
                "information_need": "核实当前比赛中尚未消除的不确定性。"} for action_id in ids]}


def test_protocol_is_closed_schema_and_preserves_numerical_solver_boundary():
    Draft202012Validator.check_schema(CatalogDiscoveryProtocol.schema)
    assert CatalogDiscoveryProtocol.version == "semantic-scenario-discovery-v2"
    assert "METADATA ONLY" in CatalogDiscoveryProtocol.prompt
    assert "mathematical VoI solver" in CatalogDiscoveryProtocol.prompt
    assert "fixed injury/weather/odds" in CatalogDiscoveryProtocol.prompt
    assert "NOT been shown" in CatalogDiscoveryProtocol.prompt


def test_article_after_first_eighty_can_be_nominated_without_dom_filling():
    pool = [article(i) for i in range(160)]
    response = nomination("read-143", "read-87")
    assert validate_catalog_discovery(response, pool, 8) == response
    assert len(response["article_candidates"]) == 2
    detached = validate_catalog_discovery(response, pool, 8)
    detached["article_candidates"][0]["information_need"] = "不同的需求"
    assert detached != response


def test_empty_nomination_is_valid_even_when_articles_exist():
    response = nomination()
    assert validate_catalog_discovery(response, [article()], 8) == response
    assert validate_catalog_discovery(response, [article()], 0) == response


def test_queries_and_articles_share_limit():
    response = nomination("read-0", queries=[{"query": "A B latest fixture context",
                                             "information_need": "本场未知信息"}])
    assert validate_catalog_discovery(response, [article()], 2) == response
    with pytest.raises(ValueError, match="candidate_limit"):
        validate_catalog_discovery(response, [article()], 1)


@pytest.mark.parametrize("change", [
    lambda response: response.update({"VoI": 0.2}),
    lambda response: response.update({"selected_action": "read-0"}),
    lambda response: response["article_candidates"][0].update({"url": "https://invented.test"}),
    lambda response: response["article_candidates"][0].update({"probability": 0.8}),
    lambda response: response.pop("queries"),
    lambda response: response.update({"schema": "semantic-scenario-discovery-v1"}),
])
def test_schema_rejects_uncontracted_fields_before_capability_use(change):
    response = nomination("read-0")
    change(response)
    with pytest.raises(ValueError, match="schema rejected"):
        validate_catalog_discovery(response, [article()], 8)


@pytest.mark.parametrize("ids,pool", [
    (("read-unknown",), [article()]),
    (("read-0",), [article(tool="search_then_read")]),
    (("read-0",), [article(), article()]),
    (("read-0", "read-0"), [article()]),
])
def test_unknown_wrong_tool_ambiguous_and_duplicate_ids_rejected(ids, pool):
    with pytest.raises(ValueError):
        validate_catalog_discovery(nomination(*ids), pool, 8)


@pytest.mark.parametrize("need", ["", " \t ", "x" * 2401, None])
def test_article_information_need_is_bounded_nonempty_text(need):
    response = nomination("read-0")
    response["article_candidates"][0]["information_need"] = need
    with pytest.raises(ValueError):
        validate_catalog_discovery(response, [article()], 8)


@pytest.mark.parametrize("query", ["", "  ", "query\nnext", "query\x00next", "x" * 301])
def test_queries_reuse_v1_nonempty_length_and_control_character_checks(query):
    response = nomination(queries=[{"query": query, "information_need": "本场未知信息"}])
    with pytest.raises(ValueError):
        validate_catalog_discovery(response, [], 8)


def test_queries_reuse_v1_case_whitespace_distinctness_and_need_checks():
    response = nomination(queries=[
        {"query": "Arsenal Leeds", "information_need": "需求一"},
        {"query": " arsenal   LEEDS ", "information_need": "需求二"},
    ])
    with pytest.raises(ValueError, match="distinct"):
        validate_catalog_discovery(response, [], 8)
    response["queries"] = [{"query": "Arsenal Leeds", "information_need": "  "}]
    with pytest.raises(ValueError, match="nonempty"):
        validate_catalog_discovery(response, [], 8)


def test_schema_caps_queries_and_articles_independently():
    too_many_queries = nomination(queries=[{"query": f"query {i}", "information_need": "需求"}
                                          for i in range(4)])
    too_many_articles = nomination(*(f"read-{i}" for i in range(9)))
    for response in (too_many_queries, too_many_articles):
        with pytest.raises(ValueError, match="schema rejected"):
            validate_catalog_discovery(response, [article(i) for i in range(9)], 8)


@pytest.mark.parametrize("limit", [-1, 9, 1.5, True, "8"])
def test_candidate_limit_is_a_bounded_integer(limit):
    with pytest.raises(ValueError, match="candidate_limit"):
        validate_catalog_discovery(nomination(), [], limit)


def test_catalog_has_explicit_metadata_boundary_and_never_claims_global_completeness():
    pool = [article(i) for i in range(CATALOG_LIMIT + 12)]
    pool.insert(0, article(999, tool="search_then_read"))
    result = discovery_catalog(pool)
    assert len(result["observed_articles"]) == CATALOG_LIMIT
    assert result["observed_articles"][0]["action_id"] == "read-0"
    assert result["observed_articles"][-1]["action_id"] == "read-319"
    assert result["catalog_coverage"] == {
        "available_articles": CATALOG_LIMIT + 12, "shown_articles": CATALOG_LIMIT,
        "omitted_articles": 12, "catalog_limit": CATALOG_LIMIT,
        "scope": "observed_metadata_only_no_body_or_value", "global_completeness": False,
    }
    assert discovery_catalog([])["catalog_coverage"]["global_completeness"] is False


def test_catalog_projects_only_scalar_metadata_without_body_result_or_value():
    raw = article(title="T" * 401, body="PRIVATE_BODY", result="FUTURE_RESULT",
                  gain="SECRET_VALUE", evidence={"quote": "HIDDEN_QUOTE"},
                  probabilities=[0.1, 0.2, 0.7])
    snapshot = deepcopy(raw)
    result = discovery_catalog([raw])
    item = result["observed_articles"][0]
    assert set(item) == {"action_id", "url", "title", "parent_url", "available_at"}
    assert len(item["title"]) == 400
    for secret in ("PRIVATE_BODY", "FUTURE_RESULT", "SECRET_VALUE", "HIDDEN_QUOTE"):
        assert secret not in json.dumps(result)
    assert raw == snapshot
    malformed = article(title={"body": "PRIVATE_BODY"}, parent_url=["FUTURE_RESULT"])
    assert "PRIVATE_BODY" not in json.dumps(discovery_catalog([malformed]))
    assert "FUTURE_RESULT" not in json.dumps(discovery_catalog([malformed]))


def test_unshown_article_is_rejected_when_passed_exact_visible_pool():
    pool = [article(i) for i in range(CATALOG_LIMIT + 1)]
    visible = pool[:CATALOG_LIMIT]
    with pytest.raises(ValueError, match="visible read_url"):
        validate_catalog_discovery(nomination("read-320"), visible, 8)
