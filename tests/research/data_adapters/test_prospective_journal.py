from datetime import datetime, timedelta, timezone

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_research.data_adapters.prospective_journal import JournaledProspectiveExecutor
from football_spider.storage.raw_archive import RawArchive


def setup_executor(tmp_path, *, status="OK", crash_after_archive=False, predictor_policy=None):
    now = datetime.now(timezone.utc)
    archive = RawArchive(tmp_path / "source")
    tools = OpenWebTools(archive, ["example.com"], extraction_version=3)
    action = {"action_id": "query-news", "tool": "search_web", "query": "Home Away preview"}
    other = {"action_id": "read-news", "tool": "read_url", "url": "https://example.com/news"}
    public = {"documents": [], "state": {"match_id": "spftll:match:1"}}
    fixture = {"match_id": "spftll:match:1", "competition_id": "spftll:league:1",
        "home_team_id": "spftll:team:1", "away_team_id": "spftll:team:2",
        "kickoff": (now + timedelta(hours=3)).isoformat()}

    class ExistingExecutor:
        seed_urls = ["https://example.com/news"]

        def __init__(self):
            self.tools, self.calls = tools, 0

        def __call__(self, selected, state):
            self.calls += 1
            # A test substitute for existing transport: assert durable intent
            # already exists at the I/O boundary, then use the real archive.
            pending = wrapped.journal.read(wrapped.run_id)
            assert pending["state"] == "PENDING_RETURN"
            requested = datetime.now(timezone.utc)
            prepared = tools.prepare(selected, {"task": {"seed_urls": self.seed_urls}, "documents": []})
            observation = tools.archive.save(prepared["source"], prepared["endpoint"], prepared["params"],
                b"raw response" if status == "OK" else b"", status=200 if status != "FAILED" else 503,
                observed_at=datetime.now(timezone.utc), context={"requested_at": requested.isoformat()})
            if crash_after_archive:
                raise RuntimeError("interrupted after raw commit")
            return {"status": status, "document": {"status": status, "text": "raw response"},
                    "receipt": {"observation": observation.model_dump(mode="json")}}

    executor = ExistingExecutor()
    wrapped = JournaledProspectiveExecutor(executor, root=tmp_path / "journal", fixture=fixture,
        request_limit=2, policy_arm="random_explore", policy_version="test-policy-v1",
        random_seed=125, propensity_logging=True, predictor_policy=predictor_policy)
    menu = {"step": 0, "public_before": public, "generated_actions": [action, other],
            "menu": [{"action_id": action["action_id"], "selected": True, "selection_probability": .5},
                     {"action_id": other["action_id"], "selected": False, "selection_probability": .5}]}
    return wrapped, executor, tools, archive, action, public, menu


class FrozenPredictorPolicy:
    predictor_only = True
    version = "test-policy-v1"

    def __init__(self):
        from types import SimpleNamespace
        self.predictor = SimpleNamespace(model_version="f-test-v1")

    def journal_initial_assessment(self):
        return {"status": "PREDICTOR_ONLY", "prediction": {"home": .5, "draw": .3, "away": .2},
                "candidate_expected_gains": None, "model_versions": ["f-test-v1"]}


def context_payload(kind, *, before=None, after=None):
    now = datetime.now(timezone.utc).isoformat()
    return {"schema": "runtime-context-v1", "snapshot_kind": kind,
            "state": {"prediction_cutoff": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()},
            "documents": [], "observed_at": now, "available_at": now,
            "availability_basis": "runtime_state_handoff", "prediction_cutoff": now,
            "acquisition_deadline": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "prediction_before": before, "prediction_after": after,
            "predictor_model_versions": ["f-test-v1"]}


@pytest.mark.parametrize("status", ["OK", "NO_DATA", "FAILED"])
def test_predictor_only_records_exact_runtime_predictions_and_response(tmp_path, status):
    wrapped, executor, tools, archive, action, public, menu = setup_executor(
        tmp_path, status=status, predictor_policy=FrozenPredictorPolicy())
    p = {"home": .5, "draw": .3, "away": .2}
    q = {"home": .55, "draw": .25, "away": .2} if status == "OK" else p
    initial = context_payload("initial", after=p)
    first = wrapped.observe_context(initial)
    assert first["prediction_after"] == p
    assert first["context"]["acquisition_deadline"] == initial["acquisition_deadline"]
    assert first["prediction_cutoff"] == initial["observed_at"]
    public["prediction"] = {"probabilities": [.5, .3, .2], "model_version": "f-test-v1"}
    wrapped.observe_context(context_payload("before_action", before=p))
    wrapped.observe_menu(menu)
    wrapped(action, public)
    after = context_payload("after_action", before=p, after=q)
    after["parent_attempt_id"] = "actual-attempt-1"
    snap = wrapped.observe_context(after)
    assert snap["prediction_before"] == p and snap["prediction_after"] == q
    assert snap["parent_request_id"] == "runtime-step-0"
    assert snap["parent_attempt_id"] == "actual-attempt-1"
    wrapped.observe_context(context_payload("final", after=q))
    run = wrapped.journal.read(wrapped.run_id)
    assert run["model_status"] == "PREDICTOR_ONLY"
    assert run["prediction"] == q and run["candidate_expected_gains"] is None
    assert run["intents"]["runtime-step-0"]["assessment"]["prediction"] == p
    assert run["intents"]["runtime-step-0"]["selection_probability"] == .5
    assert executor.calls == 1


def test_predictor_context_missing_forecast_is_not_filled_and_zero_requests_possible(tmp_path):
    wrapped, executor, *_ = setup_executor(tmp_path, predictor_policy=FrozenPredictorPolicy())
    snap = wrapped.observe_context(context_payload("initial"))
    assert snap["prediction_before"] is None and snap["prediction_after"] is None
    stopped = wrapped.finish("no_feasible_candidate")
    assert stopped["requests"] == 0 and executor.calls == 0


def test_predictor_model_change_rejected_before_io(tmp_path):
    policy = FrozenPredictorPolicy()
    wrapped, executor, _, _, action, public, menu = setup_executor(tmp_path, predictor_policy=policy)
    public["prediction"] = {"probabilities": [.5, .3, .2], "model_version": "f-test-v1"}
    wrapped.observe_menu(menu)
    policy.predictor.model_version = "changed-f"
    with pytest.raises(ValueError, match="change frozen model"):
        wrapped(action, public)
    assert executor.calls == 0


def test_runtime_context_must_identify_model_and_actual_cutoff(tmp_path):
    wrapped, *_ = setup_executor(tmp_path, predictor_policy=FrozenPredictorPolicy())
    payload = context_payload("initial")
    payload["predictor_model_versions"] = ["another-f"]
    with pytest.raises(ValueError, match="frozen predictor model versions"):
        wrapped.observe_context(payload)
    payload["predictor_model_versions"] = ["f-test-v1"]
    payload["prediction_cutoff"] = payload["acquisition_deadline"]
    with pytest.raises(ValueError, match="future acquisition horizon"):
        wrapped.observe_context(payload)


def test_after_context_cannot_replace_acquisition_before_prediction(tmp_path):
    wrapped, _, _, _, action, public, menu = setup_executor(tmp_path, predictor_policy=FrozenPredictorPolicy())
    public["prediction"] = {"probabilities": [.5, .3, .2], "model_version": "f-test-v1"}
    wrapped.observe_menu(menu)
    wrapped(action, public)
    with pytest.raises(ValueError, match="actual acquisition assessment"):
        wrapped.observe_context(context_payload("after_action", before={"home": .6, "draw": .2, "away": .2}))


@pytest.mark.parametrize("status,processing", [("OK", "DATA_AVAILABLE"), ("NO_DATA", "NO_DATA"),
                                               ("FAILED", "REQUIRES_REVIEW")])
def test_existing_journal_preserves_actual_menu_and_response(tmp_path, status, processing):
    wrapped, executor, tools, archive, action, public, menu = setup_executor(tmp_path, status=status)
    wrapped.observe_menu(menu)
    returned = wrapped(action, public)
    assert returned["status"] == status
    assert returned["receipt"]["acquisition_journal"]["run_id"] == wrapped.run_id
    assert returned["receipt"]["acquisition_journal"]["request_id"] == "runtime-step-0"
    journal = wrapped.journal.read(wrapped.run_id)
    intent = journal["intents"]["runtime-step-0"]
    assert intent["candidates"] == ["query-news", "read-news"]
    assert intent["candidate_metadata"][1]["selected"] is False
    assert intent["selection_probability"] == .5
    assert intent["selection_basis"] == "uniform_exploration"
    assert intent["assessment"]["status"] == "NOT_CONFIGURED"
    assert intent["assessment"]["prediction"] is None
    assert journal["processing"]["runtime-step-0"]["status"] == processing
    assert journal["budget"]["charged_requests"] == 1
    assert tools.archive is archive
    observations = archive.observations()
    assert len(observations) == 1
    assert observations[0].context["journal_run_id"] == wrapped.run_id
    assert observations[0].context["archive_save_timing_basis"] == "measured_local_utc_archive_save_start"
    capture_started = datetime.fromisoformat(observations[0].context["archive_save_started_at"])
    save_timing = returned["receipt"]["archive_save_timing"]
    capture_returned = datetime.fromisoformat(save_timing["archive_save_returned_at"])
    assert observations[0].observed_at <= capture_started <= capture_returned <= datetime.now(timezone.utc)
    assert save_timing["observation_id"] == str(observations[0].id)
    assert save_timing["raw_sha256"] == observations[0].content_hash
    assert save_timing["persistence_boundary"] == "return_observed_fsync_not_instrumented"
    assert "archive_save_returned_at" not in observations[0].context
    assert "document_admitted" not in returned
    stopped = wrapped.finish("request_allowance_exhausted")
    assert stopped["state"] == "STOPPED"
    assert executor.calls == 1


def test_menu_must_be_actual_before_executor_is_invoked(tmp_path):
    wrapped, executor, tools, archive, action, public, menu = setup_executor(tmp_path)
    with pytest.raises(ValueError, match="matching actual runtime menu"):
        wrapped(action, public)
    assert executor.calls == 0
    assert archive.observations() == []


def test_crash_after_raw_commit_recovers_without_second_request(tmp_path):
    wrapped, executor, tools, archive, action, public, menu = setup_executor(tmp_path, crash_after_archive=True)
    wrapped.observe_menu(menu)
    with pytest.raises(RuntimeError, match="interrupted after raw commit"):
        wrapped(action, public)
    journal = wrapped.journal.read(wrapped.run_id)
    assert journal["pending_request_id"] is None
    assert journal["budget"]["charged_requests"] == 1
    assert tools.archive is archive
    with pytest.raises(RuntimeError, match="replay or reconcile"):
        wrapped(action, public)
    assert executor.calls == 1
    assert len(archive.observations()) == 1
    assert "archive_save_returned_at" not in archive.observations()[0].context
