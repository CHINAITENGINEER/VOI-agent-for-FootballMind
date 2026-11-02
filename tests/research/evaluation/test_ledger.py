"""Mechanism fixtures only; these scores are never real match evaluation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.baselines.api import Prediction
from football_research.evaluation import ledger as module
from football_research.evaluation.ledger import ForecastLedger
from football_research.evaluation.ledger_cli import main


NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
KICKOFF = NOW + timedelta(hours=2)


@pytest.fixture
def clock(monkeypatch):
    clock = [NOW]
    monkeypatch.setattr(module, "_now", lambda: clock[0])
    return clock


def prediction(**changes):
    return {"match_id": "test:match:1", "prediction_cutoff": NOW.isoformat(),
            "home_win_probability": .6, "draw_probability": .25, "away_win_probability": .15,
            "model_version": "test-frequency-v1", "dataset_version": "test-dataset-v1",
            "input_manifest": {"test_fixture": True}, "information_mode": "operational", **changes}


def result(**changes):
    return {"match_id": "test:match:1", "information_mode": "operational", "status": "final",
            "home_goals": 2, "away_goals": 0, "score_period": "regulation",
            "source": "test-source", "source_reference": "test-only:observation:1", "revision_id": "v1",
            "available_at": (KICKOFF + timedelta(hours=2)).isoformat(),
            "observed_at": (KICKOFF + timedelta(hours=2)).isoformat(), **changes}


class Service:
    def predict(self, fixture, *, prediction_cutoff, features):
        return Prediction(**prediction(match_id=fixture["match_id"], prediction_cutoff=prediction_cutoff))


def fixture():
    return {"match_id": "test:match:1", "kickoff": KICKOFF.isoformat()}


def after_match(clock):
    clock[0] = KICKOFF + timedelta(hours=4)


def test_replay_import_issued_at_is_unverified_and_idempotent(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    row = prediction(issued_at=(NOW - timedelta(hours=1)).isoformat())
    first = ledger.import_forecast(row, kickoff=KICKOFF)
    path = tmp_path / "forecasts" / (first["forecast_id"] + ".json")
    before = path.read_bytes()
    clock[0] += timedelta(minutes=1)
    second = ledger.import_forecast(row, kickoff=KICKOFF, source_reference="another path")
    assert second["status"] == "duplicate"
    assert first["record"] == second["record"]
    assert first["record"]["issued_at"] is None
    assert first["record"]["registration_kind"] == "replay"
    assert first["record"]["claimed_issued_at_unverified"] == row["issued_at"]
    assert path.read_bytes() == before


def test_prospective_generation_has_real_issue_clock_and_pending(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    row = ledger.issue_forecast(Service(), fixture())
    record = row["record"]
    assert record["registration_kind"] == "prospective"
    assert record["issued_at"] == NOW.isoformat()
    assert record["generation_started_at"] == NOW.isoformat()
    assert record["prediction"]["prediction_cutoff"] == NOW.isoformat()
    assert ledger.score()["status_counts"] == {"pending": 1}
    after_match(clock)
    assert ledger.score()["status_counts"] == {"no_label": 1}


def test_issued_at_is_completion_clock_not_requested_cutoff(tmp_path, clock):
    class TimedService(Service):
        def predict(self, *args, **kwargs):
            prediction = super().predict(*args, **kwargs)
            clock[0] += timedelta(seconds=2)
            return prediction
    ledger = ForecastLedger(tmp_path)
    record = ledger.issue_forecast(TimedService(), fixture(), prediction_cutoff=NOW - timedelta(seconds=30))["record"]
    assert record["prediction"]["prediction_cutoff"] == (NOW - timedelta(seconds=30)).isoformat()
    assert record["issued_at"] == (NOW + timedelta(seconds=2)).isoformat()
    assert record["generation_started_at"] == NOW.isoformat()


def test_preexisting_schedule_is_not_final_label_availability(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    scheduled = ledger.settle_result(result(status="scheduled", home_goals=None, away_goals=None,
                       available_at=NOW.isoformat(), observed_at=NOW.isoformat()))
    after_match(clock)
    settled = ledger.settle_result(result(revision_id="final-v2", supersedes=[scheduled["label_version"]]))
    assert settled["record"]["first_available_at"] == NOW.isoformat()
    assert settled["record"]["first_final_label_available_at"] == (KICKOFF + timedelta(hours=2)).isoformat()
    assert ledger.score()["forecasts"][0]["first_final_label_available_at"] == (KICKOFF + timedelta(hours=2)).isoformat()


@pytest.mark.parametrize("cutoff", [NOW + timedelta(seconds=1), NOW - timedelta(seconds=301)])
def test_future_or_stale_cutoff_rejected(tmp_path, clock, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        ForecastLedger(tmp_path).issue_forecast(Service(), fixture(), prediction_cutoff=cutoff)


@pytest.mark.parametrize("advance", [timedelta(seconds=301), timedelta(hours=3), timedelta(seconds=-1)])
def test_delayed_or_clock_reversed_generation_rejected(tmp_path, clock, advance):
    class SlowService(Service):
        def predict(self, *args, **kwargs):
            output = super().predict(*args, **kwargs)
            clock[0] += advance
            return output
    ledger = ForecastLedger(tmp_path)
    with pytest.raises(ValueError, match="time window"):
        ledger.issue_forecast(SlowService(), fixture())
    assert ledger.forecasts() == []


def test_issue_after_kickoff_and_false_future_issued_at_rejected(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    with pytest.raises(ValueError, match="future"):
        ledger.import_forecast(prediction(issued_at=(NOW + timedelta(hours=1)).isoformat()), kickoff=KICKOFF)
    clock[0] = KICKOFF
    with pytest.raises(ValueError, match="precede kickoff"):
        ledger.issue_forecast(Service(), fixture())


def test_conflicting_slot_preserves_first_forecast(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    with pytest.raises(ValueError, match="slot conflict"):
        ledger.import_forecast(prediction(home_win_probability=.5, draw_probability=.35), kickoff=KICKOFF)
    assert ledger.forecasts()[0]["prediction"]["home_win_probability"] == .6


def test_concurrent_idempotent_registration_and_settlement(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        registrations = list(pool.map(lambda _: ledger.import_forecast(prediction(), kickoff=KICKOFF), range(20)))
    assert sum(r["status"] == "registered" for r in registrations) == 1
    after_match(clock)
    with ThreadPoolExecutor(max_workers=8) as pool:
        labels = list(pool.map(lambda _: ledger.settle_result(result()), range(20)))
    assert sum(r["status"] == "appended" for r in labels) == 1
    assert len(ledger.forecasts()) == len(ledger.results()) == 1


def test_result_revision_idempotence_hash_and_historical_score(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    before = {p.name: p.read_bytes() for p in (tmp_path / "forecasts").glob("*.json")}
    after_match(clock)
    first_at = clock[0]
    first = ledger.settle_result(result())
    scored = ledger.score()
    repeated = ledger.settle_result(result())
    assert repeated["status"] == "duplicate"
    assert ledger.score() == scored
    assert scored["status_counts"] == {"scored": 1}
    assert scored["groups"][0]["accuracy"] == 1
    clock[0] += timedelta(hours=1)
    revised = result(home_goals=0, away_goals=1, revision_id="v2", available_at=clock[0].isoformat(),
                     observed_at=clock[0].isoformat(), supersedes=[first["label_version"]], reason="test correction")
    second = ledger.settle_result(revised)
    report = ledger.score()
    assert report["groups"][0]["accuracy"] == 0
    assert report["forecasts"][0]["label_version"] == second["label_version"]
    assert len(ledger.results()) == 2
    assert ledger.score(as_of=first_at) == scored
    assert {p.name: p.read_bytes() for p in (tmp_path / "forecasts").glob("*.json")} == before


def test_conflicting_sources_and_explicit_resolution(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    first = ledger.settle_result(result())
    second = ledger.settle_result(result(source="test-other", home_goals=0, away_goals=2))
    assert ledger.score()["status_counts"] == {"conflict": 1}
    assert ledger.score()["groups"][0]["n_predictions"] == 0
    ledger.settle_result(result(revision_id="resolved-v2", supersedes=[first["label_version"], second["label_version"]]))
    assert ledger.score()["status_counts"] == {"scored": 1}


@pytest.mark.parametrize("status", ["cancelled", "postponed", "disputed", "scheduled"])
def test_nonfinal_statuses_are_not_scored(tmp_path, clock, status):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    ledger.settle_result(result(status=status, home_goals=None, away_goals=None))
    assert ledger.score()["status_counts"] == {"no_label" if status == "scheduled" else status: 1}
    assert ledger.score()["groups"][0]["n_predictions"] == 0


def test_postponed_match_new_kickoff_excludes_old_forecast(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    postponed = ledger.settle_result(result(status="postponed", home_goals=None, away_goals=None))
    clock[0] += timedelta(days=2)
    ledger.settle_result(result(kickoff=(KICKOFF + timedelta(days=1)).isoformat(),
                        available_at=clock[0].isoformat(), observed_at=clock[0].isoformat(),
                        revision_id="v2", supersedes=[postponed["label_version"]]))
    assert ledger.score()["status_counts"] == {"schedule_changed": 1}


@pytest.mark.parametrize("changes", [
    {"available_at": (NOW - timedelta(hours=1)).isoformat()},
    {"observed_at": "2999-01-01T00:00:00Z"},
    {"score_period": "extra_time"}, {"home_goals": True}, {"away_goals": -1},
    {"source_reference": None}, {"supersedes": ["missing"]},
    {"information_mode": "historical_reconstruction"},
])
def test_invalid_labels_rejected(tmp_path, clock, changes):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    with pytest.raises(ValueError):
        ledger.settle_result(result(**changes))
    assert ledger.results() == []


def test_revision_identity_reuse_rejected(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    ledger.settle_result(result())
    with pytest.raises(ValueError, match="revision identity"):
        ledger.settle_result(result(home_goals=1))


def test_modes_and_replay_prospective_scores_stay_separate(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.issue_forecast(Service(), fixture())
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    ledger.import_forecast(prediction(information_mode="historical_reconstruction"), kickoff=KICKOFF)
    after_match(clock)
    ledger.settle_result(result())
    report = ledger.score()
    assert report["status_counts"] == {"scored": 2, "no_label": 1}
    ledger.settle_result(result(information_mode="historical_reconstruction", availability_assumption="test kickoff+2h"))
    report = ledger.score()
    assert len(report["groups"]) == 3
    assert all(group["n_predictions"] == 1 for group in report["groups"])
    assert "log_loss" not in report


@pytest.mark.parametrize("folder", ["forecasts", "results"])
def test_tampering_stops_scoring(tmp_path, clock, folder):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    ledger.settle_result(result())
    path = next((tmp_path / folder).glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["source_reference"] = "tamper"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ledger.score()


def test_cli_register_settle_score(tmp_path, clock, capsys):
    path = tmp_path / "ledger"
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    predictions.write_text(json.dumps({**prediction(), "kickoff": KICKOFF.isoformat()}) + "\n", encoding="utf-8")
    main(["register-forecasts", "--ledger", str(path), "--predictions", str(predictions)])
    after_match(clock)
    labels.write_text(json.dumps(result()) + "\n", encoding="utf-8")
    main(["settle-forecasts", "--ledger", str(path), "--results", str(labels)])
    main(["score-forecasts", "--ledger", str(path), "--output", str(report)])
    assert json.loads(report.read_text(encoding="utf-8"))["status_counts"] == {"scored": 1}
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_batch_revision_can_reference_earlier_batch_version_and_repeat(tmp_path, clock):
    reference = ForecastLedger(tmp_path / "reference")
    batch = ForecastLedger(tmp_path / "batch")
    for ledger in (reference, batch):
        ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    original = result()
    first = reference.settle_result(original)
    correction = result(revision_id="v2", home_goals=0, away_goals=1,
                        supersedes=[first["label_version"]], reason="batch correction")
    reference.settle_result(correction)
    operations = batch.settle_results([original, original, correction])
    assert [operation["status"] for operation in operations] == ["appended", "duplicate", "appended"]
    assert operations[0]["label_version"] == first["label_version"]
    assert operations[2]["record"]["result"]["supersedes"] == [operations[0]["label_version"]]
    assert batch.score() == reference.score()
    repeated = batch.settle_results([original, correction])
    assert all(operation["status"] == "duplicate" for operation in repeated)
    assert batch.score() == reference.score()


def test_batch_reads_and_verifies_each_snapshot_once(tmp_path, clock, monkeypatch):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    calls = []
    original_records = ledger._records
    def tracked(folder):
        calls.append(folder)
        return original_records(folder)
    monkeypatch.setattr(ledger, "_records", tracked)
    operations = ledger.settle_results(result(revision_id=f"v{i}") for i in range(20))
    assert len(operations) == 20
    assert calls == ["forecasts", "results"]


def test_batch_validation_failure_preserves_prefix_and_retry_recovers(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    with pytest.raises(ValueError, match="nonnegative"):
        ledger.settle_results([result(), result(revision_id="v2", home_goals=-1)])
    versions = ledger.results()
    assert len(versions) == 1
    frozen = (tmp_path / "results" / (versions[0]["label_version"] + ".json")).read_bytes()
    correction = result(revision_id="v2", home_goals=0, away_goals=1, supersedes=[versions[0]["label_version"]])
    operations = ledger.settle_results([result(), correction])
    assert [operation["status"] for operation in operations] == ["duplicate", "appended"]
    assert (tmp_path / "results" / (versions[0]["label_version"] + ".json")).read_bytes() == frozen
    assert ledger.score()["groups"][0]["accuracy"] == 0


def test_batch_write_failure_can_resume_without_rewriting_committed_files(tmp_path, clock, monkeypatch):
    ledger = ForecastLedger(tmp_path)
    ledger.import_forecast(prediction(), kickoff=KICKOFF)
    after_match(clock)
    original_write = module._write_new
    count = [0]
    def fail_second(path, record):
        count[0] += 1
        if count[0] == 2:
            raise OSError("test-only interrupted write")
        original_write(path, record)
    monkeypatch.setattr(module, "_write_new", fail_second)
    inputs = [result(), result(source="corroborating-source")]
    with pytest.raises(OSError, match="interrupted"):
        ledger.settle_results(inputs)
    assert len(ledger.results()) == 1
    monkeypatch.setattr(module, "_write_new", original_write)
    operations = ledger.settle_results(inputs)
    assert [operation["status"] for operation in operations] == ["duplicate", "appended"]
    assert len(ledger.results()) == 2
    assert ledger.score()["status_counts"] == {"scored": 1}


def test_empty_batch_does_not_write_labels(tmp_path, clock):
    ledger = ForecastLedger(tmp_path)
    assert ledger.settle_results([]) == []
    assert ledger.results() == []
