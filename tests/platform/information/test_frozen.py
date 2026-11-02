"""Offline information-boundary tests; all fixtures here are synthetic, not data.

The tiny publication follows the real history-export shape. It deliberately
contains labels/forecast/private metadata so tests detect accidental exposure.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from football_platform.information.costs import ExperimentConfig, simulated_cost
from football_platform.information.frozen import FrozenEvidenceEnvironment, completed_bundle


MATCH_ID = "synthetic:match:target"
CUTOFF = "2025-10-10T12:00:00+00:00"
GROUPS = {"strength", "recent_form", "rolling_xg"}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def synthetic_records(*, operational=False):
    """Never written to any production raw archive or repository."""
    observation = "2025-10-09T12:00:00+00:00" if operational else "2026-09-20T12:00:00+00:00"
    ingestion = "2025-10-09T12:01:00+00:00" if operational else "2026-09-22T12:01:00+00:00"
    mode = "operational" if operational else "historical_reconstruction"
    match = {
        "match_id": MATCH_ID,
        "source": "understat", "source_id": "synthetic-target",
        "competition_id": "synthetic:competition", "season": "2025",
        "home_team_id": "synthetic:home", "away_team_id": "synthetic:away",
        "home_team_name": "Synthetic Home", "away_team_name": "Synthetic Away",
        "kickoff": "2025-10-11T12:00:00+00:00",
        "fixture_available_at": "2025-10-04T12:00:00+00:00",
        "observed_at": observation, "ingested_at": ingestion,
        "first_observed_at": observation, "observation_id": "synthetic-observation",
        "version_id": "synthetic-target-version", "data_version": "private-target-digest",
        "status": "finished", "score_period": "regulation",
        "home_goals": 7, "away_goals": 6,
        "result_available_at": "2025-10-11T15:00:00+00:00",
        "private_label": "LABEL_CANARY_DO_NOT_EXPOSE",
        "payload_path": "PRIVATE_PATH_CANARY/raw/synthetic.gz",
    }
    history = [
        {
            "match_id": f"synthetic:match:{side}:past",
            "version_id": f"synthetic-version-{side}",
            "kickoff": "2025-10-01T12:00:00+00:00",
            "result_available_at": "2025-10-01T15:00:00+00:00",
            "observed_at": observation, "ingested_at": ingestion,
            "observation_id": "synthetic-observation",
        }
        for side in ("home", "away")
    ]
    values = {
        "elo": {"home": 1520.0, "away": 1480.0, "difference": 40.0},
        "home_form": {"version_ids": ["synthetic-version-home"], "rest_days": 10.0},
        "away_form": {"version_ids": ["synthetic-version-away"], "rest_days": 10.0},
        "forecast": {"home": 0.999, "private_canary": "FORECAST_CANARY"},
        "target_xg": 9999.0,
    }
    for side in ("home", "away"):
        for field, value in {"goals_for": 2.0, "goals_against": 1.0,
                             "points": 3.0, "matches": 1,
                             "xg_for": 1.8, "xg_against": 0.9,
                             "xg_difference": 0.9}.items():
            values[f"{side}_rolling_{field}"] = value
    snapshot = {
        "match_id": MATCH_ID, "prediction_cutoff": CUTOFF,
        "information_mode": mode, "feature_version": "prematch-season-v2.0",
        "features": values,
        "manifest_hash": "PRIVATE_FULL_MANIFEST_HASH_CANARY",
        "manifest": {
            "history_inputs": history,
            "xg_inputs": [
                {"match_id": item["match_id"], "statistic_id": f"synthetic-xg-{index}",
                 "xg_available_at": "2025-10-02T12:00:00+00:00",
                 "provider": "understat", "definition_version": "understat-match-xg-v1"}
                for index, item in enumerate(history)
            ],
            "private_label": "LABEL_CANARY_DO_NOT_EXPOSE",
            "assumptions": {"result_delay_hours": 3, "xg_delay_hours": 24},
        },
    }
    return {"matches": [match], "features": [snapshot], "quality": []}


def publish(tmp_path, records=None, name="synthetic-complete"):
    records = deepcopy(records if records is not None else synthetic_records())
    root = tmp_path / name
    root.mkdir()
    manifest = {"schema": "footballmind.platform.history-export.v1", "files": {},
                "synthetic_test_fixture": True, "private_evidence": "MANIFEST_CANARY"}
    for key, rows in records.items():
        data = b"".join(json_bytes(row) + b"\n" for row in rows)
        path = f"{key}.jsonl"
        (root / path).write_bytes(data)
        manifest["files"][key] = {"path": path, "sha256": hashlib.sha256(data).hexdigest(),
                                  "row_count": len(rows)}
    commit_manifest(root, manifest)
    return root


def commit_manifest(root, manifest):
    body = json_bytes(manifest)
    (root / "manifest.json").write_bytes(body)
    (root / "COMPLETE.json").write_bytes(json_bytes({"manifest_sha256": hashlib.sha256(body).hexdigest()}))


def config(*, bindings=None, budget=None, mode="simulated_reveal", seed=7):
    bindings = bindings or {group: [group] for group in sorted(GROUPS)}
    costs = {} if mode == "cache_views" else {
        action: simulated_cost(money=0.02, requests=1, latency_ms=20, tokens=3)
        for action in bindings
    }
    return ExperimentConfig(mode=mode, bindings=bindings, costs=costs,
                            budget=budget or {}, seed=seed)


def environment(tmp_path, *, records=None, settings=None, perturbations=None):
    root = publish(tmp_path, records)
    env = FrozenEvidenceEnvironment.from_export(root, settings or config(), perturbations=perturbations)
    mode = (records or synthetic_records())["features"][0]["information_mode"]
    env.start(MATCH_ID, CUTOFF, mode)
    return env


def assert_zero_cost(cost):
    assert all(getattr(cost, part).amount == 0 for part in ("money", "requests", "latency_ms", "tokens"))


def test_completed_export_loads_small_real_shape(tmp_path):
    root = publish(tmp_path)
    manifest, rows = completed_bundle(root)
    assert manifest["synthetic_test_fixture"] is True
    assert len(rows["matches"]) == len(rows["features"]) == 1
    assert rows["matches"][0]["home_goals"] == 7


@pytest.mark.parametrize("damage, message", [
    ("marker", "manifest mismatch"), ("file", "file mismatch"),
    ("row_count", "row count mismatch"), ("schema", "Unsupported completed export"),
])
def test_completed_export_rejects_corruption(tmp_path, damage, message):
    root = publish(tmp_path)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if damage == "marker":
        (root / "COMPLETE.json").write_text('{"manifest_sha256":"wrong"}')
    elif damage == "file":
        with (root / "features.jsonl").open("ab") as target:
            target.write(b"\n")
    else:
        if damage == "row_count":
            manifest["files"]["features"]["row_count"] += 1
        else:
            manifest["schema"] = "unrecognized.schema"
        commit_manifest(root, manifest)
    with pytest.raises(ValueError, match=message):
        FrozenEvidenceEnvironment.from_export(root)


def test_completed_export_rejects_path_escape_even_with_valid_hash(tmp_path):
    root = publish(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes((root / "matches.jsonl").read_bytes())
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"]["matches"]["path"] = "../outside.jsonl"
    commit_manifest(root, manifest)
    with pytest.raises(ValueError, match="outside publication"):
        completed_bundle(root)


def test_completed_marker_is_required(tmp_path):
    root = publish(tmp_path)
    (root / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        completed_bundle(root)


def test_initial_catalog_and_state_are_allow_listed(tmp_path):
    env = environment(tmp_path)
    state = env.visible_state()
    assert state.groups == {}
    assert state.mask == dict.fromkeys(GROUPS, "UNACQUIRED")
    assert state.fixture.home_team_name == "Synthetic Home"
    data = json.dumps({"state": state.model_dump(mode="json"),
                       "catalog": [s.model_dump(mode="json") for s in env.catalog()]})
    for private in ("CANARY", "home_goals", "away_goals", "target_xg", "forecast", "payload_path"):
        assert private not in data
    unsupported = {a.action_id for a in env.catalog() if a.capability_status == "NOT_IMPLEMENTED"}
    assert unsupported == {"unavailable.weather", "unavailable.injuries"}


def test_three_groups_reveal_only_requested_values(tmp_path):
    env = environment(tmp_path)
    for index, group in enumerate(("strength", "recent_form", "rolling_xg"), 1):
        result = env.acquire(group, f"request-{index}")
        assert result.status == "ACQUIRED"
        assert set(result.revealed_groups) == {group}
        state = env.visible_state()
        assert len(state.groups) == index
        assert state.mask[group] == "OBSERVED"
        assert result.revealed_groups[group].confidence is None
        assert result.revealed_groups[group].times.feature_available_at <= state.prediction_cutoff
    assert env.visible_state().groups["rolling_xg"].values["provider"] == "understat"


def test_target_label_and_unrevealed_xg_do_not_influence_public_prefix(tmp_path):
    original = synthetic_records()
    poisoned = deepcopy(original)
    poisoned["matches"][0].update(home_goals=0, away_goals=0, status="cancelled",
                                  data_version="different-target-private-version")
    poisoned["features"][0]["features"].update(home_rolling_xg_for=5.7, target_xg=10000)
    poisoned["features"][0]["manifest_hash"] = "different-full-hidden-hash"
    env_a = FrozenEvidenceEnvironment.from_export(publish(tmp_path, original, "a"), config())
    env_b = FrozenEvidenceEnvironment.from_export(publish(tmp_path, poisoned, "b"), config())
    assert env_a.start(MATCH_ID, CUTOFF) == env_b.start(MATCH_ID, CUTOFF)
    assert env_a.catalog() == env_b.catalog()
    assert env_a.acquire("strength", "same-request") == env_b.acquire("strength", "same-request")
    assert env_a.visible_state() == env_b.visible_state()
    assert env_a.acquire("rolling_xg", "xg") != env_b.acquire("rolling_xg", "xg")


def test_public_outputs_are_detached(tmp_path):
    env = environment(tmp_path)
    initial = env.visible_state()
    initial.base_prior["home"] = 1
    initial.mask["rolling_xg"] = "OBSERVED"
    initial.fixture.home_team_name = "poison"
    catalog = env.catalog()
    catalog[0].group_ids.append("poison")
    catalog[0].expected_cost.money.amount = 999
    result = env.acquire("strength", "one")
    result.revealed_groups["strength"].values["home_elo"] = -1
    result.charged_cost.money.amount = 999
    state = env.visible_state()
    assert state.base_prior["home"] == pytest.approx(1 / 3)
    assert state.fixture.home_team_name == "Synthetic Home"
    assert state.mask["rolling_xg"] == "UNACQUIRED"
    assert state.groups["strength"].values["home_elo"] == 1520
    assert state.budget.spent["money"] == pytest.approx(0.02)
    assert all("poison" not in entry.group_ids for entry in env.catalog())


@pytest.mark.parametrize("mutation, action", [
    ("result_after_cutoff", "strength"), ("target_result", "strength"),
    ("future_kickoff", "strength"), ("xg_after_cutoff", "rolling_xg"),
])
def test_corrupt_future_lineage_is_quarantined_not_revealed(tmp_path, mutation, action):
    rows = synthetic_records()
    manifest = rows["features"][0]["manifest"]
    if mutation == "result_after_cutoff":
        manifest["history_inputs"][0]["result_available_at"] = "2025-10-10T12:00:01+00:00"
    elif mutation == "target_result":
        manifest["history_inputs"][0]["match_id"] = MATCH_ID
    elif mutation == "future_kickoff":
        manifest["history_inputs"][0]["kickoff"] = "2025-10-12T12:00:00+00:00"
    else:
        manifest["xg_inputs"][0]["xg_available_at"] = "2025-10-10T12:00:01+00:00"
    env = environment(tmp_path, records=rows)
    result = env.acquire(action, "contract-failure")
    assert result.status == "BAD_DATA"
    assert result.revealed_groups == {}
    assert env.visible_state().mask[action] == "FAILED"
    assert result.charged_cost.requests.amount == 1
    assert result.synthetic_perturbation is False


@pytest.mark.parametrize("foreign_match", [MATCH_ID, "synthetic:unlinked-match"])
def test_xg_lineage_must_link_to_non_target_history(tmp_path, foreign_match):
    rows = synthetic_records()
    rows["features"][0]["manifest"]["xg_inputs"][0]["match_id"] = foreign_match
    env = environment(tmp_path, records=rows)
    result = env.acquire("rolling_xg", "xg-contract")
    assert result.status == "BAD_DATA"
    assert result.revealed_groups == {}


def test_operational_uses_actual_observed_and_ingested_time(tmp_path):
    rows = synthetic_records(operational=True)
    env = environment(tmp_path, records=rows)
    result = env.acquire("strength", "operational")
    assert result.status == "ACQUIRED"
    assert result.revealed_groups["strength"].times.operational_available_at <= env.visible_state().prediction_cutoff


@pytest.mark.parametrize("field", ["observed_at", "ingested_at"])
def test_operational_future_observation_or_ingestion_is_hidden(tmp_path, field):
    rows = synthetic_records(operational=True)
    rows["features"][0]["manifest"]["history_inputs"][0][field] = "2025-10-10T12:00:01+00:00"
    env = environment(tmp_path, records=rows)
    result = env.acquire("strength", "operational-bad")
    assert result.status == "BAD_DATA"
    assert not result.revealed_groups


def test_operational_fixture_not_available_rejects_start(tmp_path):
    rows = synthetic_records(operational=True)
    rows["matches"][0]["ingested_at"] = "2025-10-10T12:00:01+00:00"
    env = FrozenEvidenceEnvironment.from_export(publish(tmp_path, rows), config())
    with pytest.raises(ValueError, match="Fixture unavailable"):
        env.start(MATCH_ID, CUTOFF, "operational")


def test_same_request_is_idempotent_and_cannot_change_action(tmp_path):
    env = environment(tmp_path)
    first = env.acquire("strength", "one")
    saved = env.visible_state()
    replay = env.acquire("strength", "one")
    assert replay.repeated_request is True
    assert replay.attempts == 0
    assert replay.sequence == first.sequence
    assert replay.revealed_groups == first.revealed_groups
    assert_zero_cost(replay.charged_cost)
    assert env.visible_state() == saved
    with pytest.raises(ValueError, match="another action"):
        env.acquire("rolling_xg", "one")
    assert env.visible_state() == saved


def test_another_request_for_observed_group_is_free_cache_hit(tmp_path):
    env = environment(tmp_path)
    env.acquire("strength", "first")
    result = env.acquire("strength", "new-request")
    assert result.status == "CACHE_HIT"
    assert result.content_changed is False
    assert result.attempts == 0
    assert_zero_cost(result.charged_cost)
    assert env.visible_state().budget.spent["money"] == pytest.approx(0.02)
    assert len(env.visible_state().history) == 2


def test_multi_group_binding_is_one_charge(tmp_path):
    settings = config(bindings={"results": ["strength", "recent_form"], "xg": ["rolling_xg"]})
    env = environment(tmp_path, settings=settings)
    result = env.acquire("results", "bound")
    assert set(result.revealed_groups) == {"strength", "recent_form"}
    assert result.charged_cost.requests.amount == 1
    assert result.charged_cost.money.amount == pytest.approx(0.02)
    assert env.visible_state().mask["rolling_xg"] == "UNACQUIRED"


def test_source_bundle_reveals_all_shared_response_groups_once(tmp_path):
    env = environment(tmp_path, settings=config(mode="source_bundle", bindings={"understat": sorted(GROUPS)}))
    result = env.acquire("understat", "whole-response")
    assert set(result.revealed_groups) == GROUPS
    assert result.charged_cost.requests.amount == 1
    assert_zero_cost(env.acquire("understat", "cached-response").charged_cost)


@pytest.mark.parametrize("outcome, mask", [
    ("NO_DATA", "SOURCE_MISSING"), ("TIMEOUT", "FAILED"),
    ("BAD_DATA", "FAILED"), ("STALE", "STALE"), ("UNCHANGED", "UNACQUIRED"),
])
def test_explicit_faults_have_visible_masks_and_keep_spent_cost(tmp_path, outcome, mask):
    env = environment(tmp_path, perturbations={"strength": [outcome]})
    result = env.acquire("strength", "fault")
    assert result.status == outcome
    assert result.synthetic_perturbation is True
    assert result.revealed_groups == {}
    assert result.content_changed is False
    assert result.charged_cost.money.amount == pytest.approx(0.02)
    assert result.charged_cost.requests.amount == 1
    assert result.charged_cost.latency_ms.amount == 20
    assert result.charged_cost.tokens.amount == 3
    assert env.visible_state().mask["strength"] == mask
    assert env.visible_state().budget.spent["requests"] == 1
    assert_zero_cost(env.acquire("strength", "fault").charged_cost)
    recovered = env.acquire("strength", "retry-new-operation")
    assert recovered.status == "ACQUIRED"
    assert recovered.synthetic_perturbation is False
    assert env.visible_state().budget.spent["requests"] == 2


def test_missing_xg_remains_missing_not_zero(tmp_path):
    rows = synthetic_records()
    for side in ("home", "away"):
        for key in ("xg_for", "xg_against", "xg_difference"):
            rows["features"][0]["features"][f"{side}_rolling_{key}"] = None
    env = environment(tmp_path, records=rows)
    result = env.acquire("rolling_xg", "missing")
    assert result.status == "NO_DATA"
    assert not result.revealed_groups
    assert env.visible_state().mask["rolling_xg"] == "SOURCE_MISSING"
    assert result.charged_cost.requests.amount == 1


def test_partial_data_retains_null_and_missing_reason(tmp_path):
    rows = synthetic_records()
    rows["features"][0]["features"]["home_rolling_xg_for"] = None
    env = environment(tmp_path, records=rows)
    evidence = env.acquire("rolling_xg", "partial").revealed_groups["rolling_xg"]
    assert evidence.quality == "PARTIAL"
    assert evidence.values["home_rolling_xg_for"] is None
    assert "home_rolling_xg_for" in evidence.missing_fields


@pytest.mark.parametrize("dimension, limit", [("money", 0.019), ("requests", 0), ("latency_ms", 19), ("tokens", 2)])
def test_budget_rejects_before_cost_or_projection(tmp_path, dimension, limit):
    env = environment(tmp_path, settings=config(budget={dimension: limit}))
    result = env.acquire("strength", "over-budget")
    assert result.status == "BUDGET_EXCEEDED"
    assert result.attempts == 0
    assert result.revealed_groups == {}
    assert_zero_cost(result.charged_cost)
    assert env.visible_state().mask["strength"] == "UNACQUIRED"
    assert all(value == 0 for value in env.visible_state().budget.spent.values())


def test_exact_budget_allows_one_action_but_cache_and_unsupported_are_free(tmp_path):
    env = environment(tmp_path, settings=config(budget={"money": 0.02, "requests": 1}))
    assert env.acquire("strength", "first").status == "ACQUIRED"
    assert env.acquire("rolling_xg", "blocked").status == "BUDGET_EXCEEDED"
    assert env.acquire("strength", "cache").status == "CACHE_HIT"
    unavailable = env.acquire("unavailable.weather", "not-connected")
    assert unavailable.status == "UNAVAILABLE"
    assert_zero_cost(unavailable.charged_cost)
    assert "weather" not in env.visible_state().groups


def test_existing_offline_cache_is_not_charged_as_external_http(tmp_path):
    env = environment(tmp_path, settings=config(mode="cache_views"))
    result = env.acquire("strength", "local-cache")
    assert result.revealed_groups["strength"].cache_reused is True
    assert_zero_cost(result.charged_cost)
    assert all(a.cost_scope == "cached_projection" for a in env.catalog())


def test_same_config_inputs_and_seed_replay_identically(tmp_path):
    root = publish(tmp_path)
    traces = []
    for _ in range(2):
        env = FrozenEvidenceEnvironment.from_export(root, config(seed=42), perturbations={"rolling_xg": ["TIMEOUT"]})
        traces.append([env.start(MATCH_ID, CUTOFF).model_dump(mode="json")])
        for action, request in [("strength", "one"), ("rolling_xg", "two"), ("rolling_xg", "three"), ("recent_form", "four")]:
            traces[-1].append(env.acquire(action, request).model_dump(mode="json"))
        traces[-1].append(env.visible_state().model_dump(mode="json"))
    assert json_bytes(traces[0]) == json_bytes(traces[1])


def test_restore_replays_state_and_idempotency_without_refunding(tmp_path):
    root = publish(tmp_path)
    env = FrozenEvidenceEnvironment.from_export(root, config(), perturbations={"rolling_xg": ["TIMEOUT"]})
    env.start(MATCH_ID, CUTOFF)
    env.acquire("strength", "one")
    env.acquire("rolling_xg", "failure")
    state = env.visible_state()
    restored = FrozenEvidenceEnvironment.from_export(root, config(), perturbations={"rolling_xg": ["TIMEOUT"]})
    assert restored.restore(state.model_dump(mode="json")) == state
    assert_zero_cost(restored.acquire("strength", "one").charged_cost)
    assert restored.visible_state() == state


@pytest.mark.parametrize("tamper", ["values", "spent", "mask", "history", "fixture"])
def test_restore_rejects_tampered_public_state(tmp_path, tamper):
    env = environment(tmp_path)
    env.acquire("strength", "one")
    saved = env.visible_state().model_dump(mode="json")
    if tamper == "values":
        saved["groups"]["strength"]["values"]["home_elo"] = 9000
    elif tamper == "spent":
        saved["budget"]["spent"]["money"] = 0
    elif tamper == "mask":
        saved["mask"]["rolling_xg"] = "OBSERVED"
    elif tamper == "history":
        saved["history"][0]["charged_cost"]["money"]["amount"] = 0
    else:
        saved["fixture"]["home_team_name"] = "different identity"
    with pytest.raises(ValueError, match="deterministic replay"):
        env.restore(saved)


@pytest.mark.parametrize("rpc_request", [
    {"op": "raw", "path": "private.gz"}, {"op": "manifest"}, {"op": "labels"},
    {"op": "state", "include_hidden": True}, {"op": "catalog", "sql": "SELECT *"},
    {"op": "acquire", "action_id": "strength", "request_id": "a", "raw_path": "private"},
])
def test_json_boundary_rejects_raw_or_hidden_access(tmp_path, rpc_request):
    env = environment(tmp_path)
    before = env.visible_state()
    with pytest.raises(ValueError, match="Unknown public operation or parameters"):
        env.handle(rpc_request)
    assert env.visible_state() == before


def test_json_boundary_is_executable_and_serializable(tmp_path):
    env = FrozenEvidenceEnvironment.from_export(publish(tmp_path), config())
    start = env.handle({"op": "start", "match_id": MATCH_ID, "prediction_cutoff": CUTOFF})
    result = env.handle({"op": "acquire", "action_id": "recent_form", "request_id": "json-rpc"})
    assert start["groups"] == {}
    assert set(result["revealed_groups"]) == {"recent_form"}
    json.dumps(env.handle({"op": "catalog"}))
    assert env.handle({"op": "state"})["mask"]["strength"] == "UNACQUIRED"


def test_naive_cutoff_and_unknown_snapshot_are_rejected(tmp_path):
    env = FrozenEvidenceEnvironment.from_export(publish(tmp_path), config())
    with pytest.raises(ValueError, match="Explicit timezone"):
        env.start(MATCH_ID, "2025-10-10T12:00:00")
    with pytest.raises(ValueError, match="No committed snapshot"):
        env.start("synthetic:missing", CUTOFF)


def test_ambiguous_snapshots_are_rejected(tmp_path):
    rows = synthetic_records()
    rows["features"].append(deepcopy(rows["features"][0]))
    with pytest.raises(ValueError, match="Ambiguous feature snapshot"):
        FrozenEvidenceEnvironment.from_export(publish(tmp_path, rows), config())


def test_snapshot_at_or_after_kickoff_is_not_a_prematch_episode(tmp_path):
    rows = synthetic_records()
    rows["matches"][0]["kickoff"] = CUTOFF
    env = FrozenEvidenceEnvironment.from_export(publish(tmp_path, rows), config())
    with pytest.raises(ValueError, match="Prematch cutoff"):
        env.start(MATCH_ID, CUTOFF)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_become_bad_data_without_value_disclosure(tmp_path, bad_value):
    rows = synthetic_records()
    rows["features"][0]["features"]["elo"]["home"] = bad_value
    env = environment(tmp_path, records=rows)
    result = env.acquire("strength", "nonfinite")
    assert result.status == "BAD_DATA"
    assert result.revealed_groups == {}
    assert result.charged_cost.requests.amount == 1


def test_failed_overlap_binding_does_not_erase_previously_acquired_group(tmp_path):
    settings = config(bindings={"strength": ["strength"], "results": ["strength", "recent_form"]})
    env = environment(tmp_path, settings=settings, perturbations={"results": ["TIMEOUT"]})
    strength = env.acquire("strength", "good").revealed_groups["strength"]
    result = env.acquire("results", "failed-bound-response")
    assert result.status == "TIMEOUT"
    state = env.visible_state()
    assert state.groups["strength"] == strength
    assert state.mask["strength"] == "OBSERVED"
    assert state.mask["recent_form"] == "FAILED"
    assert state.budget.spent["requests"] == 2


def test_unobserved_source_times_are_not_fabricated(tmp_path):
    env = environment(tmp_path)
    evidence = env.acquire("strength", "time-evidence").revealed_groups["strength"]
    assert evidence.times.source_published_at is None
    assert evidence.times.observed_at.isoformat() == "2026-09-20T12:00:00+00:00"
    assert evidence.times.ingested_at.isoformat() == "2026-09-22T12:01:00+00:00"
    assert evidence.times.operational_available_at == evidence.times.ingested_at
    assert evidence.times.feature_available_at < evidence.times.observed_at
    assert "reconstruction" in evidence.times.basis


@pytest.mark.parametrize("request_id", ["", "x" * 201, None, 123])
def test_invalid_request_ids_leave_state_and_budget_unchanged(tmp_path, request_id):
    env = environment(tmp_path)
    before = env.visible_state()
    with pytest.raises(ValueError, match="request_id"):
        env.acquire("strength", request_id)
    assert env.visible_state() == before


def test_catalog_hides_future_fault_schedule(tmp_path):
    root = publish(tmp_path)
    plain = FrozenEvidenceEnvironment.from_export(root, config())
    perturbed = FrozenEvidenceEnvironment.from_export(root, config(), perturbations={"strength": ["NO_DATA", "TIMEOUT"]})
    assert plain.catalog() == perturbed.catalog()
    assert plain.start(MATCH_ID, CUTOFF) == perturbed.start(MATCH_ID, CUTOFF)
    assert plain.acquire("strength", "first").status == "ACQUIRED"
    assert perturbed.acquire("strength", "first").status == "NO_DATA"


def test_rejected_restore_is_atomic_and_keeps_cost_receipts_and_fault_cursor(tmp_path):
    env = environment(tmp_path, perturbations={"rolling_xg": ["TIMEOUT"]})
    env.acquire("strength", "first")
    env.acquire("rolling_xg", "failed-once")
    before = env.visible_state()
    corrupted = before.model_dump(mode="json")
    # A truncated journal plus mismatched visible data must not reset a running
    # episode to an earlier budget/attempt cursor when recovery is rejected.
    corrupted["history"].pop()
    corrupted["fixture"]["home_team_name"] = "tampered checkpoint"
    with pytest.raises(ValueError, match="deterministic replay"):
        env.restore(corrupted)
    assert env.visible_state() == before
    replay = env.acquire("rolling_xg", "failed-once")
    assert replay.status == "TIMEOUT"
    assert replay.repeated_request is True
    assert_zero_cost(replay.charged_cost)
    assert env.acquire("rolling_xg", "retry-after-restore-rejection").status == "ACQUIRED"


def test_restore_failure_mid_replay_keeps_previous_episode(tmp_path):
    env = environment(tmp_path)
    env.acquire("strength", "one")
    before = env.visible_state()
    corrupted = before.model_dump(mode="json")
    corrupted["history"][0]["action_id"] = "not-in-catalog"
    with pytest.raises(ValueError):
        env.restore(corrupted)
    assert env.visible_state() == before


def test_published_after_cutoff_is_rejected_even_if_observed_times_claim_earlier(tmp_path):
    rows = synthetic_records(operational=True)
    rows["features"][0]["manifest"]["history_inputs"][0]["source_published_at"] = "2025-10-10T12:00:01+00:00"
    env = environment(tmp_path, records=rows)
    result = env.acquire("strength", "inconsistent-publication")
    assert result.status == "BAD_DATA"
    assert not result.revealed_groups
    assert env.visible_state().mask["strength"] == "FAILED"


def test_fixture_future_source_publication_is_not_available_to_operational_start(tmp_path):
    rows = synthetic_records(operational=True)
    rows["matches"][0]["source_published_at"] = "2025-10-10T12:00:01+00:00"
    env = FrozenEvidenceEnvironment.from_export(publish(tmp_path, rows), config())
    with pytest.raises(ValueError, match="Fixture unavailable"):
        env.start(MATCH_ID, CUTOFF, "operational")


@pytest.mark.parametrize("unlinked_version", ["synthetic-target-version", "synthetic:missing-version"])
def test_recent_form_cannot_silently_drop_unlinked_input_versions(tmp_path, unlinked_version):
    rows = synthetic_records()
    rows["features"][0]["features"]["home_form"]["version_ids"] = [unlinked_version]
    env = environment(tmp_path, records=rows)
    result = env.acquire("recent_form", "unlinked-form")
    assert result.status == "BAD_DATA"
    assert not result.revealed_groups
