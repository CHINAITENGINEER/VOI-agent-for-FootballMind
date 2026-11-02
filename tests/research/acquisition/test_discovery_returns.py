"""Mechanism and contract tests; these fixtures are not real football data."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_research.acquisition.discovery_returns import DiscoveryReturnModel
from football_research.acquisition.text_returns import TextReturnModel, _digest


P = [1/3, 1/3, 1/3]
TEXTS = ["amber cedar forward amber cedar", "cobalt birch midfield cobalt birch", "violet willow defense violet willow"]


def action(name, cost=.02):
    return {"action_id": name, "tool": "read_url", "url": "https://club.example/"+name,
            "available": True, "cost_status": "simulated", "cost": {"money": cost, "requests": 1.}}


INDEX, ARTICLE = action("news"), action("report")


def record(mid, y, a, response, history=None, menu=None):
    return {"match_id": mid, "outcome": y, "prior_probabilities": P,
            "prediction_cutoff": "2025-01-01T10:00:00Z", "result_available_at": "2025-01-02T10:00:00Z",
            "action": a, "response": response, "history": history or [],
            "discovered_actions": [] if menu is None else menu, "menu_complete": True}


def rows():
    result = []
    for y in range(3):
        for j in range(30):
            mid = f"train-{y}-{j}"
            response = ({"status": "FAILED", "kind": "unknown", "text": ""} if j >= 27 else
                        {"status": "OK", "kind": "index", "text": "Club news"})
            menu = [ARTICLE] if j < 24 else []
            result.append(record(mid, y, INDEX, response, menu=menu))
            if menu:
                z = y if j < 18 else (y+1+j % 2) % 3
                result.append(record(mid, y, ARTICLE, {"status": "OK", "kind": "article", "text": TEXTS[z]},
                                     [{"action": INDEX, "response": response}]))
    return result


def fitted(data=None):
    data = rows() if data is None else data
    text = TextReturnModel().fit(data, as_of="2025-02-01T00:00:00Z", dataset_version="constructed-only",
        evidence_origin="constructed_mechanism", min_matches=3., min_class_matches=.5,
        action_clusters=2, text_clusters=3)
    discovery = DiscoveryReturnModel(text).fit(data, as_of="2025-02-01T00:00:00Z",
        dataset_version="constructed-only", evidence_origin="constructed_mechanism")
    return text, discovery


@pytest.fixture(scope="module")
def models():
    return fitted()


def initial(text):
    return text.start(P, "heldout", "2025-03-01T00:00:00Z")


def estimate(models, *, request=INDEX, menu=None, money=.1, requests=3, state=None):
    text, model = models
    return model.enumerate(initial(text) if state is None else state, request,
        budget={"limits": {"money": money, "requests": requests}, "spent": {}},
        lambda_cost={"money": 1.}, current_actions=[request] if menu is None else menu)


def test_neutral_discovery_has_learned_positive_continuation(models):
    result = estimate(models)
    assert result["status"] == "supported"
    assert result["response_enumeration"]["value"]["gross_evsi"] == pytest.approx(0, abs=1e-12)
    assert result["response_enumeration"]["value"]["net_voi"] == pytest.approx(-.02)
    assert result["value"]["net_voi"] > 0
    assert result["expected_future_cost"] == pytest.approx(.8*.02)
    assert result["expected_future_cost_vector"]["requests"] == pytest.approx(.8)
    assert result["value"]["cost"] == pytest.approx(.02+.8*.02)
    assert result["value"]["gross_evsi"] == pytest.approx(result["immediate_evsi"]+result["expected_future_evsi"])
    assert result["value"]["net_voi"] == pytest.approx(result["value"]["gross_evsi"]-result["value"]["cost"])
    assert sum(b["probability"] for b in result["continuation_branches"]) == pytest.approx(1)
    # Search is not rewarded on failed or observed-empty returns.
    failures = [b for b in result["continuation_branches"] if b["response_token"] == "failure"]
    assert failures and all(b["selected_child_id"] is None for b in failures)
    empty = [b for b in result["continuation_branches"] if not b["discovered_request_classes"]]
    assert sum(b["probability"] for b in empty) == pytest.approx(.2)
    assert all(b["continuation_value"] == 0 for b in empty)


def test_planning_hypotheses_do_not_mutate_observed_state_or_current_menu(models):
    text, model = models
    state, menu = initial(text), [deepcopy(INDEX)]
    before, old_menu = deepcopy(state), deepcopy(menu)
    result = estimate(models, state=state, menu=menu)
    assert state == before and menu == old_menu
    assert "https://club.example/report" not in json.dumps(result)
    after = text.update(state, INDEX, {"status": "OK", "kind": "index", "text": "Club news"},
                        decision=result["response_enumeration"])
    assert after["prediction_after"] == pytest.approx(P)
    assert len(after["state"]["history"]) == 1


@pytest.mark.parametrize("money,requests", [(.02, 5), (.1, 1)])
def test_both_resource_budgets_charge_both_queries(models, money, requests):
    result = estimate(models, money=money, requests=requests)
    assert result["status"] == "supported"
    assert result["expected_future_evsi"] == 0
    assert result["expected_future_cost"] == 0
    assert result["value"]["net_voi"] == pytest.approx(-.02)


def test_known_child_actual_cost_takes_priority_over_training_prototype(models):
    costly = deepcopy(ARTICLE)
    costly["cost"]["money"] = 2.
    result = estimate(models, menu=[INDEX, costly], money=.1)
    assert result["expected_future_evsi"] == 0
    assert result["value"]["net_voi"] == pytest.approx(-.02)
    unavailable = {**deepcopy(ARTICLE), "available": False}
    result = estimate(models, menu=[INDEX, unavailable])
    assert result["expected_future_evsi"] == 0


def test_unknown_children_do_not_hide_supported_positive_policy(models):
    unknown = {**action("unseen"), "url": "https://different.example/fresh"}
    result = estimate(models, menu=[INDEX, unknown])
    assert result["status"] == "supported" and result["value"]["net_voi"] > 0
    assert result["unresolved_alternatives"]
    assert unknown["action_id"] in result["unresolved_child_ids"]
    expensive = {**deepcopy(INDEX), "cost": {"money": 3., "requests": 1.}}
    result = estimate(models, request=expensive, menu=[expensive, unknown], money=4.)
    assert result["status"] == "unknown"
    assert result["value"]["net_voi"] is None
    assert result["supported_subpolicy_value"]["net_voi"] < 0


def test_missing_parent_transition_mass_is_unknown_not_renormalized(models):
    text, _ = models
    subset = [r for r in rows() if r["response"]["status"] != "FAILED"]
    model = DiscoveryReturnModel(text).fit(subset, as_of="2025-02-01T00:00:00Z",
        dataset_version="missing-failure-menus", evidence_origin="constructed_mechanism")
    result = estimate((text, model))
    assert result["status"] == "unknown"
    assert result["reason"] == "insufficient_discovery_transition_support"
    assert result["value"]["net_voi"] is None


def test_duplicate_training_records_do_not_inflate_match_support(models):
    text, model = models
    repeated = DiscoveryReturnModel(text).fit(rows()*4, as_of="2025-02-01T00:00:00Z",
        dataset_version="duplicates", evidence_origin="constructed_mechanism")
    assert len(repeated.records) == len(model.records)
    first, second = estimate(models), estimate((text, repeated))
    assert first["continuation_branches"] == second["continuation_branches"]
    assert first["value"]["net_voi"] == second["value"]["net_voi"]
    assert max(b["transition_support"]["max_match_weight"] for b in first["continuation_branches"]) <= 1+1e-12


def test_repeating_one_observation_does_not_shift_request_prototype():
    data = rows()
    replacements = {}
    for row in data:
        if row["discovered_actions"]:
            replacement = {**ARTICLE, "url": ARTICLE["url"]+"/"+row["match_id"]}
            replacements[row["match_id"]] = replacement
            row["discovered_actions"] = [replacement]
        elif row["action"]["action_id"] == ARTICLE["action_id"]:
            row["action"] = replacements[row["match_id"]]
    text, discovery = fitted(data)
    duplicated = DiscoveryReturnModel(text).fit(data+[data[0]]*100, as_of="2025-02-01T00:00:00Z",
        dataset_version="duplicates", evidence_origin="constructed_mechanism")
    assert duplicated.prototypes == discovery.prototypes
    assert estimate((text, duplicated))["continuation_branches"] == estimate((text, discovery))["continuation_branches"]


def test_transition_records_bind_to_frozen_eligible_text_records(models):
    text, _ = models
    data = rows()
    data[0]["outcome"] = 2
    with pytest.raises(ValueError, match="bound"):
        DiscoveryReturnModel(text).fit(data, as_of="2025-02-01T00:00:00Z",
            dataset_version="tampered-label", evidence_origin="constructed_mechanism")


@pytest.mark.parametrize("change", ["unknown_menu", "future_menu", "future_body"])
def test_unobserved_or_future_information_cannot_enter_menu_law(models, change):
    text, _ = models
    data = rows()
    if change == "unknown_menu":
        data[0].pop("menu_complete")
    elif change == "future_menu":
        data[0]["menu_observed_at"] = "2025-01-03T00:00:00Z"
    else:
        data[0]["discovered_actions"] = [{**ARTICLE, "response": {"text": "future secret"}}]
    with pytest.raises(ValueError):
        DiscoveryReturnModel(text).fit(data, as_of="2025-02-01T00:00:00Z",
            dataset_version="bad-menu", evidence_origin="constructed_mechanism")


def test_unknown_sources_and_oov_never_get_a_search_bonus(models):
    unknown = {**INDEX, "url": "https://other.example/news"}
    assert estimate(models, request=unknown)["value"]["net_voi"] is None
    oov = {**INDEX, "url": "https://club.example/xenolith"}
    assert estimate(models, request=oov)["value"]["net_voi"] is None


def test_zero_request_cost_cannot_manufacture_a_free_second_http_query(models):
    free = {**deepcopy(ARTICLE), "cost": {"money": .02, "requests": 0}}
    with pytest.raises(ValueError, match="at least one physical request"):
        estimate(models, menu=[INDEX, free], requests=1)
    with pytest.raises(ValueError, match="at least one physical request"):
        estimate(models, request=free)
    text, _ = models
    data = rows()
    data[0]["discovered_actions"] = [free]
    with pytest.raises(ValueError, match="at least one physical request"):
        DiscoveryReturnModel(text).fit(data, as_of="2025-02-01T00:00:00Z",
            dataset_version="bad-cost", evidence_origin="constructed_mechanism")


def test_request_class_avoids_unique_training_url_identifiers_but_cannot_be_observed():
    # Identifiers are unrelated to outcomes. Only training vocabulary/centers
    # are used; the held-out identifier is never available during fitting.
    data = []
    context = {"status": "OK", "kind": "index", "text": "News index"}
    for i in range(180):
        child = {**ARTICLE, "url": "https://article.example/bulletin/"+sha256(str(i).encode()).hexdigest()[:24]}
        mid, y = f"unique-{i}", i % 3
        data.append(record(mid, y, INDEX, context, menu=[child]))
        data.append(record(mid, y, child, {"status": "OK", "kind": "article", "text": TEXTS[y]},
                           [{"action": INDEX, "response": context}]))
    text = TextReturnModel().fit(data, as_of="2025-02-01T00:00:00Z", dataset_version="unique-training-urls",
        evidence_origin="constructed_mechanism", min_matches=3, min_class_matches=.5,
        action_clusters=1, text_clusters=3)
    model = DiscoveryReturnModel(text).fit(data, as_of="2025-02-01T00:00:00Z",
        dataset_version="unique-training-urls", evidence_origin="constructed_mechanism")
    state = text.update(initial(text), INDEX, context)["state"]
    historic = data[1]["action"]
    assert text.enumerate(state, historic)["status"] == "unknown"
    class_plan = text.enumerate_hypothetical_request_class(state, historic)
    assert class_plan["status"] == "supported"
    assert class_plan["support"]["matching_matches"] == 180
    assert class_plan["planning_only"]
    assert class_plan["value"]["gross_evsi"] > 0
    new = {**historic, "url": "https://article.example/bulletin/never-seen-opaque-id"}
    concrete = text.enumerate(state, new)
    assert concrete["status"] == "supported"
    with pytest.raises(ValueError, match="hypothetical request-class"):
        text.update(state, historic, data[1]["response"], decision=class_plan)
    assert estimate((text, model))["value"]["net_voi"] > 0


def test_save_load_exact_and_binding_is_required(models, tmp_path):
    text, model = models
    path = tmp_path/"discovery.json"
    model.save(path)
    restored = DiscoveryReturnModel.load(path, text_model=text)
    assert estimate((text, restored)) == estimate(models)
    assert restored.metadata["production_ready"] is False
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"][0]["menu"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="content mismatch"):
        DiscoveryReturnModel.load(path, text_model=text)


@pytest.mark.parametrize("schema", ["learned-discovery-two-step-v1", "learned-discovery-two-step-v2"])
def test_old_medoid_artifact_keeps_old_schema_and_roundtrips_exactly(models, tmp_path, monkeypatch, schema):
    text, model = models
    payload = deepcopy(model._payload())
    payload["schema"] = schema
    payload["metadata"].pop("request_value_mode")
    payload["metadata"]["request_abstraction"] = "frozen_training_action_cluster_and_cost_with_training_medoid"
    legacy = {**payload, "content_sha256": _digest(payload),
              "model_version": payload["schema"]+":"+_digest(payload)[:24]}
    path = tmp_path/"old.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    restored = DiscoveryReturnModel.load(path, text_model=text)
    monkeypatch.setattr(text, "enumerate_hypothetical_request_class",
                        lambda *a, **kw: pytest.fail("Old artifact cannot acquire new centroid semantics"))
    result = estimate((text, restored))
    assert result["schema"] == payload["schema"]
    assert result["value"]["net_voi"] > 0
    restored.save(tmp_path/"copy.json")
    assert path.read_bytes() == (tmp_path/"copy.json").read_bytes()
