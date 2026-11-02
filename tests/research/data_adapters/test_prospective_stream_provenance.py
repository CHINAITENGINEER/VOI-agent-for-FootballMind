from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json

import pytest

from football_platform.information.stream_validator import validate_stream
from football_research.data_adapters.prospective_stream import (
    _identity_raw_copy,
    _match_identity,
    _verified_archive_body,
    _verified_market_fixture,
    export_batch,
    export_empty_package,
)


def test_inventory_report_digest_and_actual_run_id_are_preserved(tmp_path):
    path = tmp_path / "inventory" / "run-report.json"
    path.parent.mkdir()
    payload = b'{"schema":"prospective-fixture-inventory-v1","run_id":"actual-run","available_prospective_fixtures":0}'
    path.write_bytes(payload)
    output = tmp_path / "stream"
    report = export_empty_package(output, inventory_report={
        "path": str(path), "run_id": "actual-run", "sha256": sha256(payload).hexdigest(),
    })
    assert report["inventory_report_sha256"] == sha256(payload).hexdigest()
    assert report["inventory_run_id"] == "actual-run"
    provenance = report["inventory_provenance"]
    assert provenance["hash_status"] == "verified"
    assert provenance["run_id_status"] == "matched"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "reports" / "audit-report.json").read_text(encoding="utf-8"))
    assert manifest["external_provenance"]["fixture_inventory"] == provenance
    assert manifest["report"]["inventory_provenance"] == provenance
    assert audit["inventory_provenance"] == provenance
    assert validate_stream(output)["valid"] is True
    assert path.read_bytes() == payload


def test_inventory_report_does_not_trust_provided_digest_or_run_id(tmp_path):
    path = tmp_path / "inventory.json"
    path.write_text('{"run_id":"actual"}', encoding="utf-8")
    report = export_empty_package(tmp_path / "stream", inventory_report={
        "path": str(path), "sha256": "0" * 64, "run_id": "claimed",
    })
    provenance = report["inventory_provenance"]
    assert provenance["hash_status"] == "mismatch"
    assert provenance["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert provenance["provided_sha256"] == "0" * 64
    assert provenance["run_id"] == "actual"
    assert provenance["provided_run_id"] == "claimed"
    assert provenance["run_id_status"] == "mismatch"


def test_missing_external_inventory_hash_is_not_invented(tmp_path):
    report = export_empty_package(tmp_path / "stream", inventory_report={
        "path": str(tmp_path / "missing.json"), "run_id": "claimed",
        "sha256": "0" * 64,
    })
    assert report["inventory_report_sha256"] is None
    assert report["inventory_provenance"]["hash_status"] == "unavailable"


def test_nonempty_batch_interface_preserves_inventory_provenance(tmp_path):
    # No live request is necessary to exercise the shared export interface.
    batch = tmp_path / "batch"
    batch.mkdir()
    path = batch / "inventory.json"
    path.write_text('{"run_id":"run-from-file"}', encoding="utf-8")
    report = export_batch(batch, tmp_path / "stream", inventory_report="inventory.json")
    assert report["inventory_run_id"] == "run-from-file"
    assert report["inventory_provenance"]["resolved_path"] == str(path.resolve())
    assert report["inventory_report_sha256"] == sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prior_run(tmp_path):
    from football_research.data_adapters.prospective_text import select_initializations
    from football_research.data_adapters.spftll import load_spftll_archive
    from football_spider.storage.raw_archive import RawArchive

    moment = datetime(2026, 10, 9, 0, tzinfo=timezone.utc)
    folder = tmp_path / "match-run"
    archive = RawArchive(folder / "prior-archive")
    # An earlier, unrelated response must never be used as the fixture raw.
    unrelated = archive.save("spftll", "https://source.example/empty", {}, b'{"code":0,"data":[]}',
        status=200, observed_at=moment - timedelta(minutes=2))
    row = {"matchId": 101, "leagueId": 1864, "leagueChs": "fixture league",
        "homeId": 11, "awayId": 12, "homeChs": "Home", "awayChs": "Away",
        "footMatchState": 0, "matchTime": "2026-10-10T12:00:00",
        "spfWinFoot": 2., "spfEqualFoot": 3., "spfLoseFoot": 4.}
    observation = archive.save("spftll", "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll", {},
        json.dumps({"code": 0, "data": [{"list": [row]}]}).encode(), status=200,
        observed_at=moment - timedelta(minutes=1))
    data = load_spftll_archive(archive.root / "raw", as_of=moment)
    policy = select_initializations(data, as_of=moment, limit=1)["selected"][0]
    config = {"policy_manifest": policy,
        "initial_state": {"match_id": "spftll:match:101", "base": {}}}
    return folder, archive, observation, unrelated, config


def test_fixture_raw_falls_back_only_to_verified_prior_manifest(prior_run, tmp_path):
    folder, archive, observation, unrelated, config = prior_run
    output = tmp_path / "output"
    copied, digest, observation_id = _identity_raw_copy(folder, [], output, config=config)
    assert observation_id == observation.id
    assert observation_id != unrelated.id
    assert digest == observation.content_hash
    assert _verified_archive_body(output, copied, digest) == archive.read_body(observation)
    fixture = _verified_market_fixture(config, folder)
    identity = _match_identity(config, verified_market_fixture=fixture)
    assert identity["home_team"] == "Home"
    assert identity["away_team"] == "Away"
    assert identity["competition"] == "fixture league"
    assert identity["season"] == "unknown"
    assert identity["registration_source"] == "spftll"
    assert identity["registered_at"] == config["policy_manifest"]["market_initialization"]["frozen_at"]
    assert identity["registered_at"] != fixture["fixture_available_at"]
    assert identity["registration_timing_basis"] == "verified_market_initialization.frozen_at"


@pytest.mark.parametrize("tamper", ["raw", "manifest", "different_match"])
def test_prior_fixture_raw_fails_closed_for_unbound_or_changed_inputs(prior_run, tmp_path, tamper):
    folder, archive, observation, unrelated, config = prior_run
    if tamper == "raw":
        (archive.root / observation.payload_path).write_bytes(gzip.compress(b'{"code":0,"data":[]}'))
    elif tamper == "manifest":
        config["policy_manifest"]["market_initialization"]["fixture"]["input_manifest"] = [
            f"sha256:{unrelated.content_hash}|observation:{unrelated.id}|at:{unrelated.observed_at.isoformat()}"
        ]
    else:
        config["initial_state"]["match_id"] = "spftll:match:999"
    assert _identity_raw_copy(folder, [], tmp_path / "output", config=config) == (None, None, None)
    assert _verified_market_fixture(config, folder) == {}


def test_unbound_identity_archive_does_not_supply_arbitrary_fixture_raw(prior_run, tmp_path):
    from football_spider.storage.raw_archive import RawArchive

    folder, _, _, _, _ = prior_run
    RawArchive(folder / "identity-archive").save("official", "https://official.example/fixture", {},
        b'{"fixture":"different"}', status=200)
    assert _identity_raw_copy(folder, [], tmp_path / "output") == (None, None, None)


def test_registration_uses_actual_freeze_not_earlier_source_observation():
    fixture = {"match_id": "m1", "fixture_available_at": "2026-10-09T08:13:00+00:00",
        "kickoff": "2026-10-10T11:30:00+00:00", "source": "spftll",
        "time_basis": "verified_observation_at;Asia/Shanghai_source_clock;season_unknown"}
    config = {"run_id": "run-1", "created_at": "2026-10-09T08:37:00+00:00",
        "initial_state": {"match_id": "m1"},
        "policy_manifest": {"market_initialization": {
            "fixture": fixture, "frozen_at": "2026-10-09T08:30:00+00:00"}}}
    identity = _match_identity(config, verified_market_fixture=fixture)
    assert identity["registered_at"] == "2026-10-09T08:30:00+00:00"
    assert identity["source_observed_at"] == "2026-10-09T08:13:00+00:00"
    assert identity["registration_timing_basis"] == "verified_market_initialization.frozen_at"
    assert identity["source_timezone"] == "Asia/Shanghai"
    assert identity["original_time_text"] is None
    assert identity["timing_basis"] == fixture["time_basis"]
    config["evidence_projection"] = {"match": {"registered_at": "2026-10-09T08:25:00+00:00"}}
    explicit = _match_identity(config, verified_market_fixture=fixture)
    assert explicit["registered_at"] == "2026-10-09T08:25:00+00:00"
    assert explicit["registration_timing_basis"] == "explicit_evidence_projection.match.registered_at"


def test_registration_missing_proof_stays_null_and_bound_runtime_clock_is_separate():
    config = {"created_at": "2026-10-09T08:37:00+00:00",
        "initial_state": {"match_id": "m1"},
        "evidence_projection": {"match": {"fixture_available_at": "2026-10-09T08:13:00+00:00"}}}
    missing = _match_identity(config)
    assert missing["registered_at"] is None
    assert missing["registration_timing_basis"] == "not_recorded"
    assert missing["source_observed_at"] is None
    config["run_id"] = "run-1"
    bound = _match_identity(config)
    assert bound["registered_at"] == config["created_at"]
    assert bound["registration_timing_basis"] == "bound_runtime_configuration.created_at"
    config["registered_at"] = "2026-10-09T08:00:00"  # Timezone missing: do not conceal it with a fallback.
    invalid = _match_identity(config)
    assert invalid["registered_at"] is None
    assert "invalid_timezone_aware_time" in invalid["registration_timing_basis"]


def test_export_fixture_preserves_registration_and_source_clocks_separately(prior_run, tmp_path):
    folder, archive, observation, unrelated, config = prior_run
    config["created_at"] = "2026-10-09T00:10:00+00:00"
    config["run_id"] = "run-101"
    frozen_at = config["policy_manifest"]["market_initialization"]["frozen_at"]
    fixture = config["policy_manifest"]["market_initialization"]["fixture"]
    config_path = folder / "configuration.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_before = config_path.read_bytes()
    (folder / "run.json").write_text(json.dumps({"cycles": [], "terminal_selection": {"reason": "no_candidates"}}), encoding="utf-8")
    output = tmp_path / "stream-output"
    export_batch(folder.parent, output)
    row = json.loads((output / "fixtures.jsonl").read_text(encoding="utf-8").strip())
    assert row["registered_at"] == row["fixture_registered_at"] == frozen_at
    assert row["registration_timing_basis"] == "verified_market_initialization.frozen_at"
    assert row["source_observed_at"] == fixture["fixture_available_at"]
    assert row["registered_at"] != row["source_observed_at"]
    assert row["pre_kickoff_frozen"] is True
    assert row["timing_basis"] == fixture["time_basis"]
    assert row["source_timezone"] == "Asia/Shanghai"
    assert row["original_time_text"] is None
    assert config_path.read_bytes() == config_before
