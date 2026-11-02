"""Same-predictor empirical text gain is signed, held-out and raw-replayed."""
from copy import deepcopy
import json

import pytest

from football_research.acquisition.text_returns import _digest, VERSION as TEXT_VERSION
from football_research.acquisition.text_risk import TextRiskCalibrator, VERSION, VALUE_KIND
from test_text_returns import FIRST, SECOND, P, response, fit


@pytest.fixture(scope="module")
def text_model():
    return fit()


def calibration_rows(*, negative=False, sequential=False, n=39):
    rows = []
    for i in range(n):
        y = i % 3
        token = (y + 1) % 3 if negative else y
        row = {"match_id": f"calibration-{i:03}", "outcome": y, "prior_probabilities": P,
               "prediction_cutoff": "2026-03-01T00:00:00Z", "result_available_at": "2026-03-02T00:00:00Z",
               "action": FIRST, "response": response(token), "history": []}
        rows.append(row)
        if sequential:
            rows.append({**row, "action": SECOND, "history": [{"action": FIRST, "response": response(token)}]})
    return rows


def train(text_model, records=None, **params):
    return TextRiskCalibrator(**params).fit(calibration_rows() if records is None else records,
        text_model=text_model, as_of="2026-04-01T00:00:00Z", dataset_version="synthetic-risk-heldout",
        evidence_origin="constructed_mechanism")


def state(text_model):
    return text_model.start(P, "unseen-diagnostic-match", "2026-05-01T00:00:00Z")


def test_raw_response_replay_corrects_model_evsi_with_positive_signed_gain(text_model):
    calibrator = train(text_model)
    result = calibrator.estimate(state(text_model), FIRST, cost=.02)
    empirical, value = result["empirical_estimate"], result["value"]
    assert empirical["expected_gain"] > 0
    assert value["net_voi"] == pytest.approx(empirical["expected_gain"] - .02)
    assert value["gross_evsi"] is None and value["prior_risk"] is None and value["expected_after_risk"] is None
    assert value["coherence_error"] is None
    assert empirical["value_kind"] == VALUE_KIND and empirical["stop_certificate"] is None
    assert result["model_enumeration"]["value"]["gross_evsi"] > 0
    assert calibrator.manifest["evidence_origin"] == "constructed_mechanism"
    assert calibrator.manifest["production_ready"] is False
    for row in calibrator.records:
        raw = next(item for item in calibrator.raw_records if _digest(item) == row["raw_transition_sha256"])
        s = text_model.start(raw["prior_probabilities"], raw["match_id"], raw["prediction_cutoff"])
        p = text_model.update(s, raw["action"], raw["response"])["prediction_after"]
        import math
        delta = math.log(p[raw["outcome"]]) - math.log(P[raw["outcome"]])
        assert row["gain"] == pytest.approx(delta)
        assert row["residual"] == pytest.approx(delta - row["model_evsi"])


def test_supported_bad_updates_remain_negative_even_when_kernel_evsi_positive(text_model):
    calibrator = train(text_model, calibration_rows(negative=True))
    result = calibrator.estimate(state(text_model), FIRST, cost=.03)
    assert result["model_enumeration"]["value"]["gross_evsi"] > 0
    assert result["empirical_estimate"]["expected_gain"] < 0
    assert result["value"]["status"] == "COMPUTED"
    assert result["value"]["net_voi"] < -.03
    assert all(row["gain"] < 0 for row in calibrator.records)


def test_response_tokens_in_actual_history_condition_redundancy(text_model):
    calibrator = train(text_model, calibration_rows(sequential=True))
    initial = state(text_model)
    after = text_model.update(initial, FIRST, response(0))["state"]
    result = calibrator.estimate(after, SECOND, cost=.01)
    assert result["empirical_estimate"]["expected_gain"] == pytest.approx(0, abs=1e-12)
    assert result["value"]["net_voi"] == pytest.approx(-.01, abs=1e-12)
    assert result["empirical_estimate"]["support_matches"] == 13
    assert result["empirical_estimate"]["context_key"] != calibrator.estimate(initial, FIRST)["empirical_estimate"]["context_key"]


def test_unseen_source_or_unobserved_empirical_history_remains_unknown(text_model):
    calibrator = train(text_model)
    unseen = {"tool": "read_url", "url": "https://other.example/news/first-report"}
    for s, action in ((state(text_model), unseen),
                      (text_model.update(state(text_model), FIRST, response(0))["state"], SECOND)):
        result = calibrator.estimate(s, action)
        assert result["value"]["status"] == "INELIGIBLE"
        assert result["value"]["net_voi"] is None
        assert result["empirical_estimate"]["expected_gain"] is None


def test_insufficient_support_not_smoothing_into_zero_or_positive_value(text_model):
    calibrator = train(text_model, calibration_rows(n=3))
    result = calibrator.estimate(state(text_model), FIRST)
    assert result["empirical_estimate"]["reason"] == "insufficient_empirical_matches"
    assert result["value"]["net_voi"] is None


def test_oov_actual_return_not_silently_dropped_or_trained_as_zero(text_model):
    rows = calibration_rows()
    rows[-1]["response"] = {"status": "OK", "kind": "article", "text": "xenolith quartz zephyr"}
    calibrator = train(text_model, rows)
    assert calibrator.manifest["unmodeled_records"] == 1
    unknown = next(row for row in calibrator.records if not row["eligible"])
    assert unknown["gain"] is None and unknown["residual"] is None
    result = calibrator.estimate(state(text_model), FIRST)
    assert result["empirical_estimate"]["reason"] == "incomplete_observed_response_coverage"
    assert result["value"]["net_voi"] is None


def test_supported_failure_is_a_verified_zero_update_and_consumes_history(text_model):
    rows = calibration_rows()
    for row in rows:
        row["response"] = {"status": "FAILED", "kind": "article", "text": ""}
    calibrator = train(text_model, rows)
    assert all(row["eligible"] and abs(row["gain"]) < 1e-12 for row in calibrator.records)
    assert calibrator.estimate(state(text_model), FIRST, cost=.01)["value"]["net_voi"] == pytest.approx(-.01)


def test_duplicate_transitions_do_not_inflate_weights_or_change_fitted_version(text_model):
    rows = calibration_rows(sequential=True)
    first = train(text_model, rows)
    second = train(text_model, list(reversed(rows)) + deepcopy(rows) + deepcopy(rows[:5]))
    assert first.model_version == second.model_version
    assert first.cells == second.cells
    assert len(second.raw_records) == len(rows)


@pytest.mark.parametrize("change", ["outcome", "p0", "result_time", "response", "prefix_response"])
def test_conflicting_same_match_facts_and_observations_are_rejected(text_model, change):
    rows = calibration_rows(sequential=True)
    row = deepcopy(rows[0])
    if change == "outcome": row["outcome"] = 2
    elif change == "p0": row["prior_probabilities"] = [.5, .25, .25]
    elif change == "result_time": row["result_available_at"] = "2026-03-03T00:00:00Z"
    elif change == "response": row["response"] = response(2)
    else:
        row = deepcopy(rows[1])
        row["history"][0]["response"] = response(2)
    with pytest.raises(ValueError, match="Conflicting"):
        train(text_model, rows + [row])


@pytest.mark.parametrize("kind", ["base_training_match", "cutoff_equal_base", "future_label", "late_return", "model_after_decision", "history_reverse"])
def test_frozen_predictor_and_label_response_clock_constraints(text_model, kind):
    row = deepcopy(calibration_rows(sequential=True)[1])
    if kind == "base_training_match": row["match_id"] = text_model.metadata["training_match_ids"][0]
    elif kind == "cutoff_equal_base": row["prediction_cutoff"] = text_model.metadata["training_as_of"]
    elif kind == "future_label": row["result_available_at"] = "2026-04-02T00:00:00Z"
    elif kind == "late_return": row["response"]["available_at"] = "2026-03-01T00:00:01Z"
    elif kind == "model_after_decision": row["decision_at"] = "2026-01-31T00:00:00Z"
    else:
        row["history"][0]["available_at"] = "2026-02-28T23:59:30Z"
        row["decision_at"] = "2026-02-28T23:59:00Z"
    with pytest.raises(ValueError):
        train(text_model, [row])


@pytest.mark.parametrize("field", ["delta", "residual", "model_evsi", "prediction_before", "prediction_after"])
def test_handed_in_statistics_are_not_accepted_as_raw_supervision(text_model, field):
    row = deepcopy(calibration_rows()[0])
    row[field] = 999
    with pytest.raises(ValueError, match="externally computed"):
        train(text_model, [row])


def test_state_validation_blocks_calibration_targets_and_equal_fit_cutoff_without_menu(text_model):
    calibrator = train(text_model)
    for mid, cutoff in (("calibration-000", "2026-05-01T00:00:00Z"),
                         ("fresh", calibrator.manifest["training_as_of"])):
        s = text_model.start(P, mid, cutoff)
        with pytest.raises(ValueError, match="new match strictly after"):
            calibrator.validate_state(s)


def test_just_the_action_and_already_observed_state_can_affect_estimate(text_model):
    class Hidden:
        def __deepcopy__(self, memo): raise AssertionError("Future response inspected")
        def __str__(self): raise AssertionError("Future response inspected")
    calibrator = train(text_model)
    plain = calibrator.estimate(state(text_model), FIRST)
    poisoned = calibrator.estimate(state(text_model), {**FIRST, "future_response": Hidden(), "outcome": Hidden()})
    assert poisoned == plain


def test_loss_origin_cost_and_mutated_predictor_are_not_silently_accepted(text_model):
    calibrator = train(text_model)
    with pytest.raises(ValueError, match="log_loss"):
        calibrator.estimate(state(text_model), FIRST, loss="brier")
    for value in (-1, True, float("nan")):
        with pytest.raises(ValueError): calibrator.estimate(state(text_model), FIRST, cost=value)
    with pytest.raises(ValueError, match="evidence origin"):
        TextRiskCalibrator().fit(calibration_rows(), text_model=text_model, as_of="2026-04-01T00:00:00Z",
            dataset_version="wrong-origin", evidence_origin="real_observational")
    changed = deepcopy(text_model)
    other = train(changed)
    changed.parameters["bandwidth"] += .01
    with pytest.raises(ValueError, match="content no longer matches"):
        other.validate_state(state(text_model))


def test_save_load_recomputes_same_statistics_and_exact_estimate(text_model, tmp_path):
    original = train(text_model, calibration_rows(sequential=True))
    path = tmp_path / "risk.json"
    original.save(path)
    loaded = TextRiskCalibrator.load(path, text_model=text_model)
    assert loaded.model_version == original.model_version
    assert loaded.estimate(state(text_model), FIRST, cost=.08) == original.estimate(state(text_model), FIRST, cost=.08)


def test_even_rehashed_forged_statistics_fail_raw_replay_verification(text_model, tmp_path):
    model = train(text_model)
    path = tmp_path / "risk.json"
    model.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"][0]["gain"] += 9
    payload = {key: value for key, value in data.items() if key not in {"model_version", "content_sha256"}}
    data["content_sha256"] = _digest(payload)
    data["model_version"] = VERSION + ":" + _digest(payload)[:24]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce raw"):
        TextRiskCalibrator.load(path, text_model=text_model)


def test_declared_real_origin_requires_all_actual_step_clocks(text_model):
    # A schema fixture exercises origin admission, never empirical real evidence.
    real_fixture = deepcopy(text_model)
    real_fixture.metadata["evidence_origin"] = "real_observational"
    real_fixture.model_version = TEXT_VERSION + ":" + _digest(real_fixture._payload())[:24]
    rows = calibration_rows(sequential=True)
    kwargs = dict(text_model=real_fixture, as_of="2026-04-01T00:00:00Z", dataset_version="clock-contract-only",
                  evidence_origin="real_observational")
    with pytest.raises(ValueError, match="require decision, observation and availability"):
        TextRiskCalibrator().fit(rows, **kwargs)
    for row in rows:
        for index, step in enumerate(row["history"] + [row]):
            minute = 10 + index
            step.update(decision_at=f"2026-02-28T23:{minute}:00Z", observed_at=f"2026-02-28T23:{minute}:01Z",
                        available_at=f"2026-02-28T23:{minute}:02Z")
    model = TextRiskCalibrator().fit(rows, **kwargs)
    assert model.manifest["operational_clock_eligible"] is True
    assert "not_verified_by_outcome_integer" in model.manifest["label_contract"]
    del rows[1]["history"][0]["observed_at"]
    with pytest.raises(ValueError, match="require decision, observation and availability"):
        TextRiskCalibrator().fit(rows, **kwargs)
    assert train(text_model).manifest["operational_clock_eligible"] is False
