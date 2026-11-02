"""Constructed match-only workflow integration; no real HTTP or new samples."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket
import sys
from types import ModuleType

import pytest

from football_agent import live_match_workflow as workflow
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive
from test_verified_collection import verified_inputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file() and p.name != ".lock"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Constructed workflow integration must not access network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def live_inputs(verified_inputs, monkeypatch):
    inputs = verified_inputs
    root = inputs["root"]
    settings = {"schema": workflow.SETTINGS_VERSION, "purpose": "f_train",
        "preparation_http_budget": 6, "actor_requests_per_match": 4,
        "seed": 193, "route": "direct", "settlement_delay_minutes": 130,
        "settlement_retry_minutes": 60, "settlement_max_attempts": 2,
        "task_root": str(root/"jobs"), "previous_learning_plans": []}
    settings_path = root/"settings.json"
    write(settings_path, settings)
    calls = {"preparation": [], "actor": [], "result": []}

    def prepare(query, *, output, max_http_requests, resume=False):
        calls["preparation"].append({"query": query, "resume": resume, "max_http_requests": max_http_requests})
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(root/"plan.json", output/"initializations.json")
        shutil.copyfile(root/"bindings.json", output/"identity-bundle.json")
        shutil.copytree(root/"market", output/"market-archive")
        shutil.copytree(root/"identity", output/"identity-archive")
        result = {"status": "ready", "scope": "constructed_raw_identity_fixture_no_real_HTTP",
                  "collection": {"candidate_discovery_version": 8, "news_projection_version": 5}}
        write(output/"prepare-result.json", result)
        return result

    module = ModuleType("football_research.data_adapters.live_fixture_preparation")
    module.prepare_live_fixture = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def execute(self, prepared, *, run_id, request_id):
        job = self.archive.root.parents[2]
        state, learning = read(job/"state.json"), read(job/"learning-plan.json")
        assert state["status"] == "collecting" and learning["purpose"] == "f_train"
        assert parse_time(learning["created_at"]) < datetime.now(timezone.utc)
        calls["actor"].append(deepcopy(prepared))
        now = datetime.now(timezone.utc)
        body = b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else inputs["body"]
        observation = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=200, observed_at=now, context={"requested_at": now.isoformat()})
        return {"document": self.project(observation, prepared), "observation": observation.model_dump(mode="json"),
            "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}

    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    return {**inputs, "calls": calls, "settings": settings_path, "prepare_module": module,
        "job": root/"jobs"/"case"}


def start(inputs):
    return workflow.start_match("spftll:match:101", inputs["job"], settings_path=inputs["settings"])


@pytest.mark.parametrize("candidate_version,news_version", [(7,4), (8,5)])
def test_match_only_start_uses_real_registration_collection_and_portable_replay(live_inputs, candidate_version, news_version):
    inputs = live_inputs
    prepare = inputs["prepare_module"].prepare_live_fixture
    def versioned_prepare(query, **kwargs):
        prepared = prepare(query, **kwargs)
        prepared["collection"]["candidate_discovery_version"] = candidate_version
        prepared["collection"]["news_projection_version"] = news_version
        write(Path(kwargs["output"])/"prepare-result.json", prepared)
        return prepared
    inputs["prepare_module"].prepare_live_fixture = versioned_prepare
    result = start(inputs)
    assert result["status"] == "awaiting_result"
    assert len(inputs["calls"]["preparation"]) == 1
    assert 0 < len(inputs["calls"]["actor"]) == result["actor_requests"] <= 4
    folder = inputs["job"]/result["run_directory"]
    run, config = read(folder/"run.json"), read(folder/"configuration.json")
    assert config["supervisor_metadata"]["generator"]["version"] == f"evidence-document-candidates-v{candidate_version}"
    assert config["supervisor_metadata"]["transport"]["news_discovery"]["projection_version"] == news_version
    assert config["evidence_projection"]["version"] == "match-evidence-extraction-v5"
    assert run["outcome"] is None and run["initial_probabilities"] == run["final_probabilities"]
    assert not result["realized_voi_benefit_proven"] and not result["background_worker_started"]
    assert all("purpose" not in cycle["public_before"] for cycle in run["cycles"])
    assert all("candidate_contexts" in cycle["public_before"] for cycle in run["cycles"])
    before = hashes(inputs["job"])
    assert workflow.status_match(inputs["job"])["status"] == "awaiting_result"
    repeated = workflow.resume_match(inputs["job"], settle=True)
    assert repeated["status"] == "awaiting_result" and hashes(inputs["job"]) == before
    tick = workflow.tick_tasks(inputs["job"].parent)
    assert tick["tasks_advanced"] == 0 and hashes(inputs["job"]) == before
    replay = workflow.replay_match(inputs["job"], inputs["root"]/"replay")
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert replay["raw_responses_reprojected"] == result["actor_requests"]
    assert hashes(inputs["job"]) == before
    with pytest.raises(FileExistsError):
        start(inputs)
    assert len(inputs["calls"]["actor"]) == result["actor_requests"]


@pytest.mark.parametrize("damage", [None, "missing_receipt", "changed_cycle"])
def test_completed_collection_recovery_validates_full_receipts_before_adoption(live_inputs, monkeypatch, damage):
    inputs = live_inputs
    real_adopt = workflow._adopt_collection
    def process_lost(*args, **kwargs):
        raise RuntimeError("simulated process lost after durable batch completion")
    monkeypatch.setattr(workflow, "_adopt_collection", process_lost)
    with pytest.raises(RuntimeError, match="process lost"):
        start(inputs)
    assert read(inputs["job"]/"state.json")["status"] == "collecting"
    folder = inputs["job"]/"analysis/match-101"
    count = len(inputs["calls"]["actor"])
    if damage == "missing_receipt":
        (folder/"receipt-000.json").unlink()
    elif damage == "changed_cycle":
        cycle = read(folder/"cycle-000.json")
        cycle["execution_status"] = "FORGED"
        write(folder/"cycle-000.json", cycle)
    monkeypatch.setattr(workflow, "_adopt_collection", real_adopt)
    resumed = workflow.resume_match(inputs["job"])
    assert len(inputs["calls"]["actor"]) == count
    assert resumed["status"] == ("awaiting_result" if damage is None else "reconciliation_required")


def test_ambiguous_actor_intent_without_complete_batch_never_repeats_io(live_inputs, monkeypatch):
    inputs, calls = live_inputs, []
    def interrupted(*args, **kwargs):
        calls.append(1)
        Path(kwargs["output"]).mkdir(parents=True)
        raise RuntimeError("a request may have been sent but no complete receipt is available")
    monkeypatch.setattr(workflow, "collect_prepared_batch", interrupted)
    result = start(inputs)
    assert result["status"] == "reconciliation_required" and calls == [1]
    assert workflow.resume_match(inputs["job"])["status"] == "reconciliation_required"
    assert workflow.tick_tasks(inputs["job"].parent)["tasks_advanced"] == 0
    assert calls == [1] and not inputs["calls"]["actor"]


def test_expired_preparation_cannot_issue_actor_requests(live_inputs, monkeypatch):
    inputs = live_inputs
    cutoff = parse_time(inputs["plan"]["selected"][0]["market_initialization"]["prediction_cutoff"])
    monkeypatch.setattr(workflow, "_now", lambda: cutoff+timedelta(seconds=1))
    result = start(inputs)
    assert result["status"] == "expired" and not inputs["calls"]["actor"]
    assert not (inputs["job"]/"analysis").exists()
    assert workflow.resume_match(inputs["job"])["status"] == "expired"


def test_unsupported_source_is_preserved_as_failure_without_registration(live_inputs, monkeypatch):
    inputs = live_inputs
    def unsupported(*args, output, **kwargs):
        Path(output).mkdir(parents=True)
        result = {"status": "unsupported", "reason": "official_fixture_adapter_not_registered"}
        write(Path(output)/"prepare-result.json", result)
        return result
    monkeypatch.setattr(inputs["prepare_module"], "prepare_live_fixture", unsupported)
    result = start(inputs)
    assert result["status"] == "unsupported" and result["reason"] == "official_fixture_adapter_not_registered"
    assert not (inputs["job"]/"learning-plan.json").exists()
    assert not inputs["calls"]["actor"]
    assert workflow.resume_match(inputs["job"], settle=True)["status"] == "unsupported"


def test_same_canonical_fixture_in_another_job_cannot_reset_actor_budget(live_inputs):
    inputs = live_inputs
    first = start(inputs)
    count = len(inputs["calls"]["actor"])
    repeated = workflow.start_match("spftll:match:101", inputs["job"].parent/"repeat",
        settings_path=inputs["settings"])
    assert repeated["status"] == "duplicate_fixture"
    assert len(inputs["calls"]["actor"]) == count == first["actor_requests"]
    assert not (inputs["job"].parent/"repeat/analysis").exists()


@pytest.mark.parametrize("damage", [None, "receipt_id", "extra_raw"])
def test_due_result_joins_verified_raw_score_without_changing_actor(live_inputs, monkeypatch, damage):
    inputs = live_inputs
    initial = start(inputs)
    folder = inputs["job"]/initial["run_directory"]
    before, actor_calls = hashes(folder), len(inputs["calls"]["actor"])
    due = parse_time(initial["result_due_at"])+timedelta(seconds=2)
    kickoff = parse_time(inputs["binding"]["canonical"]["kickoff"])
    monkeypatch.setattr(workflow, "_now", lambda: due)
    # The export adapter independently rejects future observations relative
    # to wall time. Advance its clock as well for this constructed future test.
    import football_research.data_adapters.settlement as settlement_module
    class SettlementClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return due.astimezone(tz) if tz is not None else due.replace(tzinfo=None)
    monkeypatch.setattr(settlement_module, "datetime", SettlementClock)

    def capture(args):
        inputs["calls"]["result"].append(list(args))
        assert args[0] == "capture-results" and args[args.index("--max-http-requests")+1] == "1"
        archive = RawArchive(Path(args[args.index("--output")+1]))
        raw = {"id": "2046001", "status": "FINISHED", "seasonYear": "2027",
            "homeTeam": {"id": "101", "internationalName": "Croatia"},
            "awayTeam": {"id": "102", "internationalName": "Spain"},
            "competition": {"id": "2014", "code": "UNL", "age": "ADULT", "sex": "MALE",
                "sportsType": "FOOTBALL", "teamCategory": "NATIONAL"},
            "round": {"mode": "GROUP", "competitionId": "2014", "seasonYear": "2027",
                "metaData": {"name": "League phase", "type": "GROUP_STANDINGS"}},
            "group": {"competitionId": "2014", "seasonYear": "2027", "teams": ["101", "102"]},
            "competitionPhase": "TOURNAMENT", "type": "GROUP_STAGE",
            "kickOffTime": {"dateTime": kickoff.isoformat()},
            "fullTimeAt": (kickoff+timedelta(minutes=100)).isoformat(),
            "score": {"regular": {"home": 0, "away": 0}, "total": {"home": 0, "away": 0}}}
        observation = archive.save("open_page", "https://match.uefa.com/v5/matches/2046001", {},
            json.dumps(raw).encode(), status=200, observed_at=due-timedelta(seconds=1),
            context={"requested_at": (due-timedelta(seconds=2)).isoformat()})
        write(archive.root/"capture-receipts.json", {"http_requests_actual": 1, "max_http_requests": 1,
            "receipts": [{"observation_id": observation.id, "raw_sha256": observation.content_hash,
                "observed_at": observation.observed_at.isoformat(), "url": observation.endpoint,
                "requested_at": (due-timedelta(seconds=2)).isoformat(), "http_attempts": 1}]})
        return 0

    monkeypatch.setattr(workflow, "capture_results", capture)
    real_adopt = workflow._adopt_settlement
    def completion_lost(*args, **kwargs):
        raise RuntimeError("simulated process loss after result receipt")
    monkeypatch.setattr(workflow, "_adopt_settlement", completion_lost)
    with pytest.raises(RuntimeError, match="after result receipt"):
        workflow.resume_match(inputs["job"], settle=True)
    assert read(inputs["job"]/"state.json")["status"] == "settling"
    destination = inputs["job"]/"results/attempt-01"
    if damage == "receipt_id":
        receipt = read(destination/"capture-receipts.json")
        receipt["receipts"][0]["observation_id"] = "00000000-0000-0000-0000-000000000000"
        write(destination/"capture-receipts.json", receipt)
    elif damage == "extra_raw":
        archive = RawArchive(destination)
        first = archive.observations()[0]
        archive.save(first.source, first.endpoint, first.params, archive.read_body(first),
            status=first.http_status, observed_at=first.observed_at, context=first.context)
    monkeypatch.setattr(workflow, "_adopt_settlement", real_adopt)
    finished = workflow.resume_match(inputs["job"], settle=True)
    if damage is not None:
        assert finished["status"] == "reconciliation_required"
        assert len(inputs["calls"]["result"]) == 1 and len(inputs["calls"]["actor"]) == actor_calls
        assert not (destination/"training-qualification.json").exists()
        assert hashes(folder) == before
        assert workflow.resume_match(inputs["job"], settle=True)["status"] == "reconciliation_required"
        assert len(inputs["calls"]["result"]) == 1
        return
    assert finished["status"] == "training_eligible"
    assert finished["result_requests"] == finished["settlement_attempts"] == 1
    assert len(inputs["calls"]["result"]) == 1 and len(inputs["calls"]["actor"]) == actor_calls
    assert hashes(folder) == before
    joined = read(inputs["job"]/finished["qualification_path"])
    assert joined["independent_result_verification"][0]["status"] == "VERIFIED"
    assert all(row["label"]["outcome"] == 1 for row in joined["dataset"]["episodes"])
    assert finished["qualified_response_rows"] == len(joined["training_records"])
    assert not finished["realized_voi_benefit_proven"]
    workflow.resume_match(inputs["job"], settle=True)
    assert len(inputs["calls"]["result"]) == 1 and hashes(folder) == before


def test_mixed_batch_retains_unsupported_case_and_resume_does_not_repeat_requests(live_inputs, monkeypatch, tmp_path_factory):
    inputs = live_inputs
    real_prepare = inputs["prepare_module"].prepare_live_fixture
    def mixed_prepare(query, *, output, max_http_requests, resume=False):
        if query != "spftll:match:202":
            return real_prepare(query, output=output, max_http_requests=max_http_requests, resume=resume)
        inputs["calls"]["preparation"].append({"query": query, "resume": resume,
            "max_http_requests": max_http_requests})
        Path(output).mkdir(parents=True)
        result = {"status": "unsupported", "reason": "unregistered_official_competition"}
        write(Path(output)/"prepare-result.json", result)
        return result
    monkeypatch.setattr(inputs["prepare_module"], "prepare_live_fixture", mixed_prepare)
    # Windows tempfile APIs still enforce MAX_PATH in this environment. Keep
    # the batch/replay fixture root short without changing archive semantics.
    batch = tmp_path_factory.mktemp("b")/"b"
    result = workflow.start_batch(["spftll:match:202", "spftll:match:101"], batch,
        settings_path=inputs["settings"])
    assert result["status"] == "incomplete" and result["attempted_tasks"] == 2
    assert result["collected_tasks"] == result["confirmed_physical_matches"] == 1
    assert [r["status"] for r in result["results"]] == ["unsupported", "awaiting_result"]
    frozen = read(batch/"batch.json")
    assert [task["query"] for task in frozen["tasks"]] == ["spftll:match:202", "spftll:match:101"]
    assert frozen["actor_request_cap"] == 8
    calls, before = deepcopy(inputs["calls"]), hashes(batch)
    again = workflow.resume_batch(batch)
    assert again == result and inputs["calls"] == calls and hashes(batch) == before
