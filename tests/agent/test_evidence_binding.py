import pytest

from football_agent.decision_value import FiniteDecisionModel, ObservationTool
from football_agent.evidence_binding import bind_candidate_evidence
from football_agent.policy_audit import audit_episode_with_platform_evidence
from football_agent.policy_replay import POLICY_AGENT, replay_policy
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.evidence_binding import bind_candidate_evidence


TARGET = "match-evidence"
CUTOFF = "2026-09-27T10:00:00Z"


def board():
    return StateQueryBoard(
        target_key=TARGET,
        prediction_cutoff=CUTOFF,
        hypotheses=[
            StateHypothesis("lineup-state", TARGET, "rotation_status", ("starters", "reserves")),
            StateHypothesis("reserve-state", TARGET, "reserve_strength", ("strong", "weak")),
            StateHypothesis("weather-state", TARGET, "weather", ("clear", "rain")),
            StateHypothesis("odds-state", TARGET, "market_state", ("stable", "moving")),
        ],
        candidates=[
            StateQueryCandidate("lineup", TARGET, CUTOFF, "rotation_status", "lineup_feed", "read", .01, action_id="lineup"),
            StateQueryCandidate("reserve", TARGET, CUTOFF, "reserve_strength", "squad_feed", "read", .01, action_id="reserve"),
            StateQueryCandidate("weather", TARGET, CUTOFF, "weather", "weather_feed", "read", .01, action_id="weather"),
            StateQueryCandidate("odds", TARGET, CUTOFF, "market_state", "odds_feed", "read", .01, action_id="odds"),
        ],
    )


def row(candidate_id, *, quality="VALIDATED", dependency=None, observed="2026-09-27T08:00:00Z",
        available="2026-09-27T08:30:00Z", **extra):
    return {
        "candidate_id": candidate_id,
        "source_id": f"source-{candidate_id}",
        "dependency_key": dependency or f"dependency-{candidate_id}",
        "quality": quality,
        "observed_at": observed,
        "operational_available_at": available,
        "evidence_refs": [f"evidence:{candidate_id}"],
        "content_hash": f"hash:{candidate_id}",
        **extra,
    }


def test_platform_times_quality_and_dependency_are_bound_to_public_statuses():
    result = bind_candidate_evidence(board(), [
        row("lineup", dependency="shared-lineup"),
        row("reserve", dependency="shared-lineup"),
        row("weather", available="2026-09-27T10:05:00Z"),
        row("odds", quality="PROVIDER_ERROR", error_class="timeout"),
    ], cutoff=CUTOFF)
    assert result.availability_by_candidate == {
        "lineup": "available",
        "reserve": "duplicate",
        "weather": "late",
        "odds": "failed",
    }
    payload = result.as_dict()
    assert payload["dependency_groups"]["shared-lineup"] == ["lineup", "reserve"]
    assert payload["bindings"][1]["reason"] == "dependency_key_already_bound"


def test_actual_evidence_group_shape_is_unwrapped_without_exposing_body():
    result = bind_candidate_evidence(board(), [{
        "candidate_id": "lineup",
        "group": {
            "group_id": "lineup-group",
            "source": "spftll",
            "quality": "PARTIAL",
            "times": {
                "observed_at": "2026-09-27T08:00:00Z",
                "operational_available_at": "2026-09-27T08:30:00Z",
            },
            "dependency_key": "spftll:lineup",
            "evidence_refs": ["archive:1"],
            "content_hash": "sha256:1",
            "values": {"lineup": "redacted"},
        },
    }], cutoff=CUTOFF)
    binding = result.bindings[0]
    assert binding.availability == "available"
    assert binding.quality == "PARTIAL"
    assert binding.operational_available_at == "2026-09-27T08:30:00+00:00"


def test_missing_rows_are_explicit_source_missing_failures():
    result = bind_candidate_evidence(board(), [], cutoff=CUTOFF)
    assert all(status == "failed" for status in result.availability_by_candidate.values())
    assert all(binding.reason == "source_missing" for binding in result.bindings)


def test_audit_integration_keeps_failed_and_late_states_unresolved():
    model = FiniteDecisionModel(
        {"a": (.8, .1, .1), "b": (.1, .1, .8)},
        {"a": .5, "b": .5},
        {"lineup": ObservationTool(("starters", "reserves"), {"a": (1, 0), "b": (0, 1)}, .01)},
    )
    episode = replay_policy(model, POLICY_AGENT, seed=5, budget=.05)
    result = audit_episode_with_platform_evidence(
        board(), episode,
        [row("lineup", available="2026-09-27T10:30:00Z"),
         row("reserve", quality="PROVIDER_ERROR", error_class="http_503")],
        cutoff=CUTOFF,
        view="agent",
    )
    assert result["status_by_candidate"]["lineup"] == "late"
    assert result["status_by_candidate"]["reserve"] == "failed"
    assert result["state_resolution_status"]["rotation_status"] == "unresolved_late"
    assert result["state_resolution_status"]["reserve_strength"] == "unresolved_failed"
    assert result["evidence_binding"]["schema_version"] == "platform-evidence-binding-v1"
    assert result["hidden_result_included"] is False


@pytest.mark.parametrize("quality", ["SOURCE_MISSING", "STALE", "INVALID", "PROVIDER_ERROR"])
def test_failure_qualities_are_kept_as_failed_availability(quality):
    result = bind_candidate_evidence(board(), [row("lineup", quality=quality)], cutoff=CUTOFF)
    assert result.availability_by_candidate["lineup"] == "failed"


def test_expired_validated_evidence_is_reclassified_as_stale_before_readiness():
    result = bind_candidate_evidence(
        board(), [row("lineup", expires_at="2026-09-27T09:00:00Z")], cutoff=CUTOFF,
    )
    binding = result.bindings[0]
    assert binding.availability == "failed"
    assert binding.quality == "STALE"
    assert binding.reason == "evidence_expired_before_prediction_cutoff"


def test_out_of_board_evidence_row_is_rejected():
    with pytest.raises(ValueError, match="outside the board"):
        bind_candidate_evidence(board(), [row("unknown")], cutoff=CUTOFF)
