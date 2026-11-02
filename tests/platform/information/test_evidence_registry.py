"""Constructed raw sources check registration/projection, not football outcomes."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

from football_platform.information.evidence_registry import CompetitionEvidenceRegistry
from football_spider.storage.raw_archive import RawArchive

FIXTURE = dict(match_id="unit-match", competition_id="unit-cup", home_team_id="a-team", away_team_id="b-team",
               kickoff="2026-09-20T12:00:00Z")
CUTOFF = "2026-09-20T11:00:00Z"
PUBLISHED = b"2026-09-19T10:00:00Z"
BODY = b'{"qualified_team":"a-team","published_at":"' + PUBLISHED + b'"}'
PUBLICATION_ANCHOR = {"byte_start": BODY.index(PUBLISHED), "byte_end": BODY.index(PUBLISHED) + len(PUBLISHED),
                      "sha256": hashlib.sha256(PUBLISHED).hexdigest()}


def at(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def make(tmp_path, *, now="2026-09-26T11:00:00Z", observed="2026-09-26T10:00:00Z"):
    archive = RawArchive(tmp_path / "archive")
    obs = archive.save("unit_provider", "https://example.test/preview", {}, BODY, status=200, observed_at=at(observed))
    registry = CompetitionEvidenceRegistry(tmp_path / "registry", archive, clock=lambda: at(now))
    return registry, obs


def register(registry, obs, **changes):
    args = dict(evidence_id="qualification-a", fixture=FIXTURE, observation=obs, document_scope="pre_match_preview",
                claim={"topic": "qualification", "value": "qualified", "team_id": "a-team"},
                anchor={"byte_start": 0, "byte_end": len(BODY), "sha256": hashlib.sha256(BODY).hexdigest()})
    args.update(changes)
    if args.get("source_published_at") is not None:
        args.setdefault("publication_time_anchor", PUBLICATION_ANCHOR)
    return registry.register(**args)


def project(registry, **changes):
    args = dict(fixture=FIXTURE, prediction_cutoff=CUTOFF, acquired_evidence_ids=["qualification-a"])
    args.update(changes)
    return registry.project(**args)


def test_old_preview_captured_now_is_not_operationally_visible_at_old_cutoff(tmp_path):
    registry, obs = make(tmp_path)
    record = register(registry, obs)
    assert record.operational_available_at == at("2026-09-26T11:00:00Z")
    result = project(registry)
    assert result.context.teams["a-team"]["qualification"].status == "unknown"
    assert result.exclusions == {"qualification-a": "not_available_at_prediction_cutoff"}
    assert result.actual_observation_times["qualification-a"]["observed_at"] == obs.observed_at.isoformat()


def test_operational_available_claim_requires_explicit_purchase_and_stays_unverified(tmp_path):
    registry, obs = make(tmp_path, now="2026-09-20T10:00:00Z", observed="2026-09-20T09:00:00Z")
    register(registry, obs)
    assert project(registry, acquired_evidence_ids=[]).context.evidence == {}
    purchased = project(registry)
    assert purchased.context.teams["a-team"]["qualification"].status == "unverified"
    assert purchased.context.teams["a-team"]["rotation"].status == "unknown"


def test_register_does_not_override_same_identity_even_with_identical_payload(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs)
    path = registry._record_path("candidate-evidence", "qualification-a")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Append-only"):
        register(registry, obs)
    assert path.read_bytes() == before
    version = register(registry, obs, evidence_id="qualification-a-v2")
    assert version.evidence_id == "qualification-a-v2"
    assert path.read_bytes() == before


def test_actual_observation_timestamp_cannot_be_backdated_by_cloning_metadata(tmp_path):
    registry, obs = make(tmp_path)
    forged = obs.model_copy(update={"observed_at": at("2026-09-19T09:00:00Z")})
    with pytest.raises(ValueError, match="immutable archive commit"):
        register(registry, forged)


def test_registry_rejects_an_observation_in_the_future_of_registration(tmp_path):
    registry, obs = make(tmp_path, now="2026-09-20T09:00:00Z")
    with pytest.raises(ValueError, match="future"):
        register(registry, obs)


@pytest.mark.parametrize("scope", ["post_match_report", "unknown"])
def test_postmatch_and_unknown_documents_enter_evaluation_quarantine(tmp_path, scope):
    registry, obs = make(tmp_path)
    record = register(registry, obs, document_scope=scope, claim=None, anchor=None)
    assert record.partition == "evaluation-quarantine"
    assert registry._record_path("evaluation-quarantine", "qualification-a").exists()
    for mode in ("operational", "historical_reconstruction"):
        result = project(registry, information_mode=mode)
        assert result.context.evidence == {}
        assert result.exclusions["qualification-a"] == "evaluation_only_document"


def test_postmatch_document_cannot_carry_a_context_claim(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="Only pre-match"):
        register(registry, obs, document_scope="post_match_report")


def test_registry_rejects_a_self_certified_verified_fact(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="does not establish verified"):
        register(registry, obs, claim={"topic": "qualification", "value": "qualified", "team_id": "a-team",
                                      "epistemic_status": "verified_fact"})


def test_claim_cannot_inject_its_own_availability_or_source_fields(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="owner-controlled"):
        register(registry, obs, claim={"topic": "qualification", "value": "qualified", "team_id": "a-team",
                                      "available_at": "2026-09-19T00:00:00Z"})


def test_unknown_publication_does_not_get_a_made_up_historical_timestamp(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs)
    assumption = {"assumption_id": "unit-assumption", "assumed_available_at": "2026-09-19T12:00:00Z",
                  "basis": "source_publication_assumed_unchanged_since_publication"}
    result = project(registry, information_mode="historical_reconstruction",
                     reconstruction_assumptions={"qualification-a": assumption})
    assert result.context.evidence == {}
    assert result.exclusions["qualification-a"] == "source_publication_timestamp_unknown"


def test_historical_reconstruction_needs_explicit_assumption_and_retains_actual_times(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs, source_published_at="2026-09-19T10:00:00Z", publication_time_basis="explicit_timestamp_in_source")
    hidden = project(registry, information_mode="historical_reconstruction")
    assert hidden.exclusions["qualification-a"] == "explicit_reconstruction_assumption_required"
    assumption = {"assumption_id": "unit-assumption", "assumed_available_at": "2026-09-19T12:00:00Z",
                  "basis": "source_publication_assumed_unchanged_since_publication"}
    result = project(registry, information_mode="historical_reconstruction",
                     reconstruction_assumptions={"qualification-a": assumption})
    assert result.information_mode == "historical_reconstruction"
    assert result.context.teams["a-team"]["qualification"].status == "unverified"
    assert result.context.evidence["qualification-a"].observed_at is None
    assert result.context.evidence["qualification-a"].available_at == at("2026-09-19T12:00:00Z")
    assert result.actual_observation_times["qualification-a"]["observed_at"] == obs.observed_at.isoformat()
    assert project(registry).context.evidence == {}


def test_assumed_historical_availability_cannot_precede_source_publication(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs, source_published_at="2026-09-19T10:00:00Z", publication_time_basis="explicit_timestamp_in_source")
    with pytest.raises(ValueError, match="precede source publication"):
        project(registry, information_mode="historical_reconstruction", reconstruction_assumptions={
            "qualification-a": {"assumption_id": "bad-assumption", "assumed_available_at": "2026-09-18T12:00:00Z",
                                "basis": "source_publication_assumed_unchanged_since_publication"}})


def test_operational_mode_rejects_reconstruction_assumptions(tmp_path):
    registry, _ = make(tmp_path)
    with pytest.raises(ValueError, match="does not accept historical"):
        project(registry, reconstruction_assumptions={"qualification-a": {}})


@pytest.mark.parametrize("anchor", [None, {"byte_start": 0, "byte_end": 999, "sha256": "0" * 64},
                                     {"byte_start": 0, "byte_end": 5, "sha256": "0" * 64}])
def test_extraction_claim_requires_a_valid_raw_byte_anchor(tmp_path, anchor):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        register(registry, obs, anchor=anchor)


def test_unpurchased_broken_registration_does_not_get_opened_or_revealed(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs)
    path = registry._record_path("candidate-evidence", "qualification-a")
    path.write_bytes(b"broken private record")
    assert project(registry, acquired_evidence_ids=[]).context.evidence == {}


def test_registration_integrity_and_raw_integrity_are_rechecked(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs)
    path = registry._record_path("candidate-evidence", "qualification-a")
    content = json.loads(path.read_bytes())
    content["record"]["registered_at"] = "2026-09-19T09:00:00Z"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        registry.read("qualification-a")


def test_acquired_record_is_tied_to_the_exact_fixture(tmp_path):
    registry, obs = make(tmp_path)
    register(registry, obs)
    with pytest.raises(ValueError, match="fixture differs"):
        project(registry, fixture={**FIXTURE, "match_id": "different-match"})


def test_registry_rejects_path_like_identity(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="inert"):
        register(registry, obs, evidence_id="../outside")


@pytest.mark.parametrize("evidence_id", ["sha256:abc123", "CON", "NUL", "PRN", "COM1", "evidence:stream"])
def test_logical_ids_use_safe_hash_filenames_including_colons_and_reserved_names(tmp_path, evidence_id):
    registry, obs = make(tmp_path)
    register(registry, obs, evidence_id=evidence_id)
    path = registry._record_path("candidate-evidence", evidence_id)
    assert path.name == hashlib.sha256(evidence_id.encode()).hexdigest() + ".json"
    assert ":" not in path.name
    assert path.parent == registry.root / "candidate-evidence"
    assert registry.read(evidence_id).evidence_id == evidence_id
    with pytest.raises(ValueError, match="Append-only"):
        register(registry, obs, evidence_id=evidence_id)


@pytest.mark.parametrize("evidence_id", ["C:\\outside", "C:outside", "../outside", "folder/record", "\\\\host\\share"])
def test_windows_and_relative_paths_never_become_registry_paths(tmp_path, evidence_id):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="inert"):
        register(registry, obs, evidence_id=evidence_id)


def test_publication_timestamp_needs_raw_source_evidence(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="raw timestamp anchor"):
        register(registry, obs, source_published_at="2026-09-19T10:00:00Z",
                 publication_time_basis="explicit_timestamp_in_source", publication_time_anchor=None)


def test_declared_publication_timestamp_must_equal_the_anchored_source_value(tmp_path):
    registry, obs = make(tmp_path)
    with pytest.raises(ValueError, match="differs from the anchored"):
        register(registry, obs, source_published_at="2026-09-18T10:00:00Z",
                 publication_time_basis="explicit_timestamp_in_source")
