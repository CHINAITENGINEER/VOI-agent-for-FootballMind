"""Same-f signed calibration on query-visible request spines, not latent labels."""
from copy import deepcopy
import json
import math

import pytest

from football_research.acquisition.neighborhood_text_risk import (
    NeighborhoodTextRiskCalibrator, load_text_risk_model, VERSION, CONTEXT_SPEC,
)
from football_research.acquisition.text_history import HistoryNeighborhoodTextReturnModel
from football_research.acquisition.text_returns import _digest
from football_research.acquisition.text_risk import TextRiskCalibrator
from test_text_history import ACTIONS, RESPONSES, P, records as original_training


@pytest.fixture(scope="module")
def text_model():
    rows = []
    for variant in range(3):
        for original in original_training():
            row = deepcopy(original)
            row["match_id"] += f"-variant-{variant}"
            for step in row["history"]+[row]:
                if step["action"] == ACTIONS[0]:
                    step["response"] = deepcopy(RESPONSES[variant])
            rows.append(row)
    return HistoryNeighborhoodTextReturnModel().fit(rows, as_of="2026-02-01T00:00:00Z",
        dataset_version="constructed-risk-spines-only", evidence_origin="constructed_mechanism",
        text_clusters=3, action_clusters=3, min_matches=12, min_class_matches=1.)


def calibration_rows(*, negative=False, n=39, order=(1, 0)):
    rows = []
    for index in range(n):
        outcome = index % 3
        history = [{"action": deepcopy(ACTIONS[step]),
                    "response": deepcopy(RESPONSES[index % 3] if step == 0 else RESPONSES[1])}
                   for step in order]
        rows.append({"match_id": f"calibration-{index:03}", "outcome": outcome,
            "prior_probabilities": P, "prediction_cutoff": "2026-03-01T00:00:00Z",
            "result_available_at": "2026-03-02T00:00:00Z", "history": history,
            "action": deepcopy(ACTIONS[2]),
            "response": deepcopy(RESPONSES[(outcome+1) % 3 if negative else outcome])})
    return rows


def train(model, rows=None, **parameters):
    return NeighborhoodTextRiskCalibrator(**parameters).fit(calibration_rows() if rows is None else rows,
        text_model=model, as_of="2026-04-01T00:00:00Z", dataset_version="heldout-public-spines",
        evidence_origin="constructed_mechanism")


def state(model, *, variant=0, order=(1, 0), prior=P):
    current = model.start(prior, "future-match", "2026-05-01T00:00:00Z")
    for step in order:
        update = model.update(current, ACTIONS[step], RESPONSES[variant] if step == 0 else RESPONSES[1])
        assert update["status"] == "updated"
        current = update["state"]
    return current


@pytest.fixture(scope="module")
def critic(text_model):
    return train(text_model)


def test_internal_signed_replay_and_public_context_pooling(text_model, critic):
    assert critic.manifest["context_spec"] == CONTEXT_SPEC
    assert critic.manifest["calibration_population"] == "prequery_supported_only"
    assert len({row["context_key"] for row in critic.records}) == 1
    assert {row["support_origin"] for row in critic.records} == {"neighbor_only"}
    assert len({text_model._history_key(state(text_model, variant=i)["history"])[1][2] for i in range(3)}) == 3
    for variant in range(3):
        result = critic.estimate(state(text_model, variant=variant), ACTIONS[2], cost=.02)
        estimate = result["empirical_estimate"]
        assert estimate["support_matches"] == 39
        assert estimate["expected_gain"] > 0
        assert result["value"]["net_voi"] == pytest.approx(estimate["expected_gain"]-.02)
        assert result["value"]["gross_evsi"] is None and estimate["stop_certificate"] is None
    for row, raw in zip(critic.records, critic.raw_records):
        current = text_model.start(raw["prior_probabilities"], raw["match_id"], raw["prediction_cutoff"])
        for step in raw["history"]:
            current = text_model.update(current, step["action"], step["response"])["state"]
        result = text_model.update(current, raw["action"], raw["response"])
        delta = math.log(result["prediction_after"][raw["outcome"]])-math.log(current["current_probabilities"][raw["outcome"]])
        assert row["gain"] == pytest.approx(delta)
        assert row["residual"] == pytest.approx(delta-row["model_evsi"])


def test_bad_actual_updates_remain_negative_even_with_positive_model_evsi(text_model):
    critic = train(text_model, calibration_rows(negative=True))
    result = critic.estimate(state(text_model), ACTIONS[2], cost=.03)
    assert result["model_enumeration"]["value"]["gross_evsi"] > 0
    assert result["empirical_estimate"]["expected_gain"] < 0
    assert result["value"]["status"] == "COMPUTED" and result["value"]["net_voi"] < -.03


def test_order_and_source_support_are_not_pooled_away(text_model, critic):
    after_ab = state(text_model, order=(0, 1))
    other = critic.estimate(after_ab, ACTIONS[2])
    assert other["empirical_estimate"]["support_origin"] == "exact_only"
    assert other["empirical_estimate"]["reason"] == "unsupported_empirical_context"
    assert other["value"]["net_voi"] is None
    unseen = {"tool": "read_url", "url": "https://other.example/gamma-gamma"}
    assert critic.estimate(state(text_model), unseen)["value"]["net_voi"] is None
    enum = text_model.enumerate(state(text_model), ACTIONS[2])
    altered = deepcopy(enum)
    altered["support"]["history_exact_transition_count"] = 1
    assert critic._public_context(state(text_model), ACTIONS[2], altered)[0] != critic._public_context(
        state(text_model), ACTIONS[2], enum)[0]


def test_confidence_local_fallback_stays_within_same_parent(text_model, critic):
    initial = state(text_model)
    high = state(text_model, prior=[.46, .27, .27])
    first, second = [critic.estimate(s, ACTIONS[2])["empirical_estimate"] for s in (initial, high)]
    assert first["context_key"] == second["context_key"]
    assert first["confidence"] != second["confidence"]
    assert second["local_weight"] == 0. and second["support_status"] == "supported_parent"
    assert second["parent_support"] == 39


def test_prequery_unknown_audited_without_poisoning_supported_population(text_model):
    rows = calibration_rows()
    unsupported = deepcopy(rows[0])
    unsupported["match_id"] = "prequery-unknown"
    unsupported["history"][1]["response"]["text"] = "unknownword xenolith zephyr"
    result = train(text_model, rows+[unsupported])
    assert result.manifest["prequery_unsupported_records"] == 1
    assert not result.incomplete_contexts
    row = next(row for row in result.records if not row["prequery_supported"])
    assert row["gain"] is None and row["residual"] is None and row["context_key"] is None
    assert result.estimate(state(text_model), ACTIONS[2])["value"]["status"] == "COMPUTED"


def test_unknown_actual_return_blocks_entire_parent_including_other_confidence(text_model):
    rows = calibration_rows()
    rows[-1]["response"]["text"] = "unknownword xenolith zephyr"
    critic = train(text_model, rows)
    assert critic.manifest["supported_query_unknown_returns"] == 1
    assert critic.manifest["prequery_supported_records"] == len(rows)
    missing = next(row for row in critic.records if not row["eligible"])
    assert missing["gain"] is None and missing["residual"] is None
    for prior in (P, [.46, .27, .27]):
        result = critic.estimate(state(text_model, prior=prior), ACTIONS[2])
        assert result["empirical_estimate"]["reason"] == "incomplete_observed_response_coverage"
        assert result["value"]["net_voi"] is None


def test_duplicates_preserve_model_version_and_match_support(text_model, critic):
    rows = calibration_rows()
    repeated = train(text_model, rows*3+list(reversed(rows)))
    assert repeated.model_version == critic.model_version
    assert repeated.cells == critic.cells
    sparse = train(text_model, calibration_rows(n=3))
    assert sparse.estimate(state(text_model), ACTIONS[2])["empirical_estimate"]["reason"] == "insufficient_empirical_matches"


@pytest.mark.parametrize("field", ["gain", "residual", "model_evsi", "prediction_after", "family", "latent", "context_key"])
def test_handed_in_statistics_or_latent_context_rejected(text_model, field):
    rows = calibration_rows()
    rows[0][field] = 123
    with pytest.raises(ValueError, match="raw transitions|raw transitions, not externally"):
        train(text_model, rows)


def test_future_action_payload_cannot_affect_prequery_estimate(text_model, critic):
    class Hidden:
        def __deepcopy__(self, memo):
            raise AssertionError("Unrevealed information inspected")
        def __str__(self):
            raise AssertionError("Unrevealed information inspected")
    plain = critic.estimate(state(text_model), ACTIONS[2])
    action = {**ACTIONS[2], "future_response": Hidden(), "outcome": Hidden(), "family": Hidden()}
    assert critic.estimate(state(text_model), action) == plain


@pytest.mark.parametrize("change", ["base_train_match", "equal_base_cutoff", "late_label", "late_history", "conflicting_label"])
def test_normalized_match_and_clock_contract_reused(text_model, change):
    rows = calibration_rows()
    if change == "base_train_match":
        rows[0]["match_id"] = text_model.metadata["training_match_ids"][0]
    elif change == "equal_base_cutoff":
        rows[0]["prediction_cutoff"] = text_model.metadata["training_as_of"]
    elif change == "late_label":
        rows[0]["result_available_at"] = "2026-04-02T00:00:00Z"
    elif change == "late_history":
        rows[0]["history"][0]["available_at"] = "2026-03-02T00:00:00Z"
    else:
        duplicate = deepcopy(rows[0])
        duplicate["outcome"] = 2
        rows.append(duplicate)
    with pytest.raises(ValueError):
        train(text_model, rows)


def test_live_state_rejects_training_reuse_and_nonfuture_cutoff(text_model, critic):
    for mid, cutoff in (("calibration-000", "2026-05-01T00:00:00Z"),
                        ("future-match", critic.manifest["training_as_of"])):
        current = text_model.start(P, mid, cutoff)
        with pytest.raises(ValueError, match="new match strictly after"):
            critic.validate_state(current)


def test_bound_predictor_and_loss_cannot_change(text_model, critic):
    with pytest.raises(ValueError, match="log_loss"):
        critic.estimate(state(text_model), ACTIONS[2], loss="brier")
    changed = deepcopy(text_model)
    changed.parameters["bandwidth"] += .01
    with pytest.raises(ValueError, match="content no longer matches"):
        train(changed)
    with pytest.raises(ValueError, match="content no longer matches"):
        TextRiskCalibrator().fit(calibration_rows(), text_model=text_model,
            as_of="2026-04-01T00:00:00Z", dataset_version="wrong-class", evidence_origin="constructed_mechanism")


def test_freeze_factory_restore_and_rederived_statistics(text_model, critic, tmp_path):
    path = tmp_path/"critic.json"
    critic.save(path)
    loaded = load_text_risk_model(path, text_model=text_model)
    assert type(loaded) is NeighborhoodTextRiskCalibrator
    assert loaded.model_version == critic.model_version
    assert loaded.estimate(state(text_model), ACTIONS[2]) == critic.estimate(state(text_model), ACTIONS[2])
    loaded.save(tmp_path/"restored.json")
    assert path.read_bytes() == (tmp_path/"restored.json").read_bytes()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"][0]["gain"] += 100
    payload = {key: value for key, value in data.items() if key not in {"model_version", "content_sha256"}}
    data.update(content_sha256=_digest(payload), model_version=VERSION+":"+_digest(payload)[:24])
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce raw"):
        load_text_risk_model(path, text_model=text_model)


def test_factory_keeps_old_class_and_artifact_semantics(tmp_path):
    from test_text_returns import fit
    from test_text_risk import train as old_train, state as old_state, FIRST
    model = fit()
    critic = old_train(model)
    path = tmp_path/"legacy.json"
    critic.save(path)
    restored = load_text_risk_model(path, text_model=model)
    assert type(restored) is TextRiskCalibrator
    assert restored.estimate(old_state(model), FIRST) == critic.estimate(old_state(model), FIRST)
    restored.save(tmp_path/"same-legacy.json")
    assert path.read_bytes() == (tmp_path/"same-legacy.json").read_bytes()
