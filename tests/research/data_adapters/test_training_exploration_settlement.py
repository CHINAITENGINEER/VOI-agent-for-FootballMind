import pytest

from football_research.data_adapters import training_exploration_settlement as settlement


def _run(monkeypatch, *, gain, selected=True, status="eligible"):
    derived = [{
        "physical_match_id": "m1", "attempt_id": "a1", "action_id": "injury",
        "policy_arm": "voi", "selection_probability": 0.75,
        "prediction_before": {"home": .5, "draw": .3, "away": .2},
        "prediction_after": {"home": .6, "draw": .2, "away": .2},
        "realized_gain": gain, "net_gain": gain, "label_status": status,
    }] if selected else []
    monkeypatch.setattr(settlement, "derive_g_candidates", lambda **kwargs: derived)
    attempts = [{
        "physical_match_id": "m1", "attempt_id": "a1", "action_id": "injury",
        "selected_by_policy": selected, "selection_probability": .75,
        "training_exploration": {
            "adapter_schema": "footballmind-g-training-exploration-adapter-v1",
            "selection_authority": "data_collection_only", "label_status": "pending_outcome",
        },
    }]
    return settlement.settle_training_exploration_rows(
        fixtures=[], discoveries=[], attempts=attempts, contexts=[], results=[]
    )


def test_positive_realized_gain_becomes_prospective_label(monkeypatch):
    rows, report = _run(monkeypatch, gain=.25)
    assert rows[0]["realized_signed_gain"] == pytest.approx(.25)
    assert rows[0]["label_status"] == "prospective_realized"
    assert rows[0]["gain_direction"] == "positive_improvement"
    assert report["rows_eligible"] == 1


def test_negative_realized_gain_is_kept_as_valid_counterexample(monkeypatch):
    rows, report = _run(monkeypatch, gain=-.2)
    assert rows[0]["realized_signed_gain"] == pytest.approx(-.2)
    assert rows[0]["gain_direction"] == "negative_harm"
    assert report["negative"] == 1


def test_unselected_action_is_null_not_negative_label(monkeypatch):
    rows, report = _run(monkeypatch, gain=None, selected=False)
    assert rows[0]["label_status"] == "unexecuted"
    assert rows[0]["realized_signed_gain"] is None
    assert report["unexecuted_null"] == 1


def test_missing_after_or_outcome_stays_pending(monkeypatch):
    rows, report = _run(monkeypatch, gain=None, status="ineligible")
    assert rows[0]["realized_signed_gain"] is None
    assert rows[0]["label_status"] == "ineligible"
    assert report["selected_without_label"] == 1

