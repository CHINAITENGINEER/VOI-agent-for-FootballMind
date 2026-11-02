import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2]/"tools/run_postread_shortcut_diagnostics.py"
spec = importlib.util.spec_from_file_location("postread_shortcut_diagnostics", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_groups_are_transitive_across_multi_source_matches():
    documents = [
        dict(physical_match_id="m1", source_domain="a", text_sha256="h1", source_url="a/1"),
        dict(physical_match_id="m2", source_domain="a", text_sha256="h2", source_url="a/2"),
        dict(physical_match_id="m2", source_domain="b", text_sha256="h3", source_url="b/2"),
        dict(physical_match_id="m3", source_domain="b", text_sha256="h4", source_url="b/3"),
        dict(physical_match_id="m4", source_domain="c", text_sha256="h5", source_url="c/4"),
    ]
    groups = list(module.source_components(documents).values())
    assert sorted(map(sorted, groups)) == [["m1", "m2", "m3"], ["m4"]]


def test_duplicate_text_links_source_components_without_domain_overlap():
    docs = [dict(physical_match_id=f"m{i}",source_domain=f"s{i}",source_url=f"s{i}/x",text_sha256="same") for i in range(3)]
    assert len(module.source_components(docs)) == 1


def test_unknown_source_is_not_silently_an_independent_cluster():
    with pytest.raises(ValueError, match="explicit"):
        module.source_components([dict(physical_match_id="m1",source_domain=None,text_sha256="x",source_url="x")])


def test_missing_match_time_cannot_be_filled_from_article_time():
    with pytest.raises(ValueError, match="No actual match date"):
        module.match_dates([dict(physical_match_id="understat:123",published_at="2026-01-01")])


def test_equal_match_weights_do_not_overweight_many_article_states():
    ids = np.array(["a"]*10+["b"])
    weights = module.match_weights(ids)
    assert np.isclose(weights[:10].sum(), weights[-1])
    losses = np.array([1.]*10+[3.])
    assert np.isclose(np.average(losses,weights=weights), module.macro(losses,ids))


def test_encoder_only_contains_supplied_training_texts():
    train = ["home player unavailable injury", "away team news goalkeeper", "match preparation schedule training"]
    enc = module.fit_encoder(train)
    before = dict(enc["vectorizer"].vocabulary_)
    module.encode(enc,["heldoutuniqueword prediction target"])
    assert enc["vectorizer"].vocabulary_ == before
    assert "heldoutuniqueword" not in before
    assert set(enc["train_text_sha256"]) == {module.text_sha(x) for x in train}


def test_features_refuse_target_fields_before_encoding():
    with pytest.raises(ValueError,match="allowlist"):
        module.designs([{"realized_gain":1.}], None, None, {}, [])


def test_actual_split_plans_keep_source_and_time_boundaries():
    if not module.DATA.exists():
        pytest.skip("Historical dataset not mounted")
    features = module.rows(module.DATA/"features.jsonl")
    samples = module.rows(module.DATA/"samples.jsonl")
    docs = module.rows(module.DATA/"documents.jsonl")
    expansion = module.read(module.EXPANSION_PLAN)
    plans,audit = module.plan_splits(features,samples,docs,expansion)
    assert len(plans) == 7
    assert plans["original_18"]["counts"] == plans["within_source_time_18"]["counts"]
    dates = audit["match_dates"]
    time_plan = plans["global_time_51"]["assignments"]
    assert max(dates[m] for m,s in time_plan.items() if s=="train") < min(dates[m] for m,s in time_plan.items() if s=="validation")
    assert max(dates[m] for m,s in time_plan.items() if s=="validation") < min(dates[m] for m,s in time_plan.items() if s=="test")
    for name,plan in plans.items():
        if name.startswith("source_holdout"):
            domains={s:{d["source_domain"] for d in docs if plan["assignments"][d["physical_match_id"]]==s} for s in module.SPLITS}
            assert not domains["train"] & domains["test"]
            assert not domains["validation"] & domains["test"]
