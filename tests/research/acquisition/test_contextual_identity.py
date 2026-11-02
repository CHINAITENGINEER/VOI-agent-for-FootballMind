from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.acquisition.contextual_identity import build_crosswalk, SourceIdentityValueEstimator
from football_research.acquisition.state import GROUP_FIELDS


def write_bundle(tmp_path, name, rows):
    path = tmp_path/name
    path.mkdir()
    (path/"id_mappings.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def mapping(entity_id, source_id="87", kind="team", source="understat"):
    return {"entity_type": kind, "source": source, "source_id": source_id, "entity_id": entity_id, "valid_to": None, "status": "source_scoped"}


def crosswalk(tmp_path):
    a = write_bundle(tmp_path, "a", [mapping("old"), mapping("opponent", "88"), mapping("competition-old", "EPL", "competition")])
    b = write_bundle(tmp_path, "b", [mapping("new"), mapping("opponent", "88"), mapping("competition-new", "EPL", "competition")])
    return build_crosswalk([a, b])


def state(day=100, home="old", competition="competition-old"):
    kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc)+timedelta(days=day)
    return {"match_id": f"m{day}", "prediction_cutoff": (kickoff-timedelta(days=1)).isoformat(),
        "information_mode": "historical_reconstruction", "base": {"home_team_id": home, "away_team_id": "opponent",
        "competition_id": competition, "kickoff": kickoff.isoformat()},
        "groups": {g: {"status": "not_acquired", "values": {}} for g in GROUP_FIELDS}}


ACTION = {"action_id": "fetch", "group_ids": ["xg"], "available": True,
          "cost": {"money": .03, "requests": 1., "latency_ms": 0., "tokens": 0.}, "cost_status": "simulated"}


def test_exact_provider_key_unifies_changed_uuid_and_never_fuzzy_matches(tmp_path):
    model = SourceIdentityValueEstimator(crosswalk(tmp_path))
    before, after = state(), state(home="new", competition="competition-new")
    assert model.project(before)[0] == model.project(after)[0]
    unknown = state(home="Liverpool")
    projected, audit = model.project(unknown)
    assert projected["base"]["home_team_id"] == "Liverpool"
    assert audit["unmapped_fields"] == ["home_team_id"]


def test_conflicting_uuid_to_two_provider_ids_is_rejected(tmp_path):
    a = write_bundle(tmp_path, "a", [mapping("same", "87")])
    b = write_bundle(tmp_path, "b", [mapping("same", "88")])
    with pytest.raises(ValueError, match="conflicting"):
        build_crosswalk([a, b])


def test_ambiguous_duplicate_provider_id_and_unsupported_provider_rejected(tmp_path):
    a = write_bundle(tmp_path, "a", [mapping("one"), mapping("two")])
    with pytest.raises(ValueError, match="Ambiguous"):
        build_crosswalk([a])
    with pytest.raises(ValueError, match="Understat only"):
        build_crosswalk([a], provider="other")


def test_unknown_other_provider_and_outcome_metadata_never_enters_crosswalk(tmp_path):
    a = write_bundle(tmp_path, "a", [{**mapping("one"), "result": "home"}, mapping("other", source="unverified")])
    result = build_crosswalk([a])
    assert result["aliases"]["team"] == {"one": "provider:understat:team:87"}
    assert "result" not in json.dumps(result)


@pytest.mark.parametrize("status", ["disputed", "rejected", None])
def test_untrusted_mapping_status_rejected(tmp_path, status):
    a = write_bundle(tmp_path, "a", [{**mapping("one"), "status": status}])
    with pytest.raises(ValueError, match="mapping status"):
        build_crosswalk([a])


def test_fit_serialization_and_equivalent_identity_inference(tmp_path):
    model = SourceIdentityValueEstimator(crosswalk(tmp_path), min_support=3)
    rows = []
    for i in range(1, 21):
        public = state(i)
        rows.append({"match_id": public["match_id"], "state": public, "kickoff": public["base"]["kickoff"],
            "action": ACTION, "delta": .2 if i%2 else -.1, "prediction_before": [.5,.3,.2],
            "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["past"],
            "result_available_at": (datetime.fromisoformat(public["base"]["kickoff"])+timedelta(hours=3)).isoformat()})
    model.fit(rows, as_of="2025-03-01T00:00:00Z", dataset_version="synthetic")
    path = tmp_path/"model.json"
    model.save(path)
    loaded = SourceIdentityValueEstimator.load(path)
    a = loaded.estimate_value(state(), [ACTION], prediction=[.5,.3,.2])
    b = loaded.estimate_value(state(home="new", competition="competition-new"), [ACTION], prediction=[.5,.3,.2])
    assert a == b
    assert a == model.estimate_value(state(), [ACTION], prediction=[.5,.3,.2])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["crosswalk"]["aliases"]["team"]["old"] = "provider:understat:team:99"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        SourceIdentityValueEstimator.load(path)
