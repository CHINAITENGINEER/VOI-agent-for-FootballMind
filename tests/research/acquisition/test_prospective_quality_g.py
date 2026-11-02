"""Boundaries for the prospective frozen-f quality-g consumer.

The two fixtures in this test are real frozen shadows.  They are deliberately
used only to verify reconstruction and exclusion behavior: neither fixture has
an independent settled outcome, so this test must never fit an empirical g.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.prospective_quality_g import (
    DATA_VERSION,
    QUERY_VERSION,
    assemble,
    fit_and_evaluate,
    prequery,
)
from football_research.data_adapters.predictor_context_pairs import _payload


ROOT = Path("var")
LIVERPOOL = ROOT / "f-entity-semantic-g-shadow-cumulative-b-20261011-pipelinev2-liverpool-v1"
PALACE = ROOT / "f-entity-semantic-g-shadow-cumulative-b-20261011-pipelinev3-palace-v1"


def _entries():
    return [
        {"stream": str(LIVERPOOL / "stream"), "registration": str(LIVERPOOL / "slot-registration.json")},
        {"stream": str(PALACE / "stream"), "registration": str(PALACE / "slot-registration.json")},
    ]


def _first_menu(root):
    line = next(line for line in (root / "stream" / "discovery.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip())
    menu = json.loads(line)
    return _payload(root / "stream", menu)


def test_real_shadows_rebuild_only_prequery_features_and_preserve_groups():
    dataset = assemble(_entries())
    assert dataset["schema"] == DATA_VERSION
    summary = dataset["summary"]
    assert summary["selected_attempts"] == 7
    assert summary["physical_matches"] == 2
    assert summary["eligible_rows"] == 0
    assert summary["prequery_feature_rows"] == 7
    assert summary["valid_predictor_pairs"] == 4
    assert summary["exclusions"]["outcome_missing"] == 7
    assert dataset["selection_allowed"] is False
    assert dataset["production_ready"] is False
    assert {row["partition"] for row in dataset["records"]} == {"train", "holdout"}
    assert {row["pipeline"] for row in dataset["records"]} == {
        "entity-market-f:680a321e570606964c39ef64d84a91a64fed5a3293f82bfa263eb31781bfe74f",
        "entity-market-f:9ae1b40f143143cc7c10cebb5ff0d0412815b1e22d5ad4a5ec8eec6072420f97",
    }
    for row in dataset["records"]:
        assert row["query"]["schema"] == QUERY_VERSION
        assert row["query"]["query_sha256"] == _digest(
            {key: value for key, value in row["query"].items() if key != "query_sha256"}
        )
        assert row["gain"] is None
        assert row["target_status"] == "ineligible"


def test_query_has_no_post_read_or_outcome_fields():
    dataset = assemble(_entries())
    serialized = json.dumps([row["query"] for row in dataset["records"]], sort_keys=True)
    assert "realized_signed_gain" not in serialized
    assert "outcome" not in serialized
    assert "responses" not in serialized
    assert "documents" not in serialized
    assert all(row["query"]["feature_origin"].startswith("derived_from_saved_pre_io")
               for row in dataset["records"])


def test_prequery_rejects_tampered_candidate_context_sidecar():
    raw = _first_menu(LIVERPOOL)
    action_id = next(action["action_id"] for action in raw["generated_actions"]
                      if action.get("tool") == "read_url")
    tampered = deepcopy(raw)
    tampered["public_before"]["candidate_contexts"]["semantics"]["contains_target_body"] = True
    with pytest.raises(ValueError, match="pre-query-only"):
        prequery(tampered, action_id)


def test_prequery_rejects_action_url_binding_changes():
    raw = _first_menu(PALACE)
    action = next(action for action in raw["generated_actions"] if action.get("tool") == "read_url")
    tampered = deepcopy(raw)
    tampered["public_before"]["candidate_contexts"]["by_action_id"][action["action_id"]]["url"] = (
        action["url"] + "?tampered=1"
    )
    with pytest.raises(ValueError, match="URL differs"):
        prequery(tampered, action["action_id"])


def test_evaluation_keeps_empty_train_and_holdout_strata_separate():
    dataset = assemble(_entries())
    models, report = fit_and_evaluate(dataset)
    assert models == {}
    assert report["models_fitted"] == 0
    assert report["selection_allowed"] is False
    assert report["production_ready"] is False
    assert set(report["strata"]) == {
        "entity-market-f:680a321e570606964c39ef64d84a91a64fed5a3293f82bfa263eb31781bfe74f",
        "entity-market-f:9ae1b40f143143cc7c10cebb5ff0d0412815b1e22d5ad4a5ec8eec6072420f97",
    }
    assert all(value["fit_status"] == "waiting_for_admitted_training_targets"
               for value in report["strata"].values())


def test_one_train_match_cannot_be_promoted_to_a_conditional_g():
    dataset = assemble(_entries())
    candidate = next(row for row in dataset["records"] if row["partition"] == "train")
    candidate = deepcopy(candidate)
    candidate.update(target_status="eligible", gain=0.12, exclusion_reason=[])
    dataset["records"] = [candidate]
    models, report = fit_and_evaluate(dataset)
    assert models == {}
    only = next(iter(report["strata"].values()))
    assert only["fit_status"] == "insufficient_training_physical_matches"
    assert only["minimum_training_physical_matches"] == 2

