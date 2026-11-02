"""As-of competition context, deliberately separate from football result labels.

The owner supplies observations and the acquisition ledger.  This module only
projects purchased evidence; it neither collects facts nor infers strength from
qualification, squad rotation, or a team's incentive to draw.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from football_spider.models.contracts import Contract
from .contracts import Fixture

CONTEXT_VERSION = "competition-context-v1"
Topic = Literal["stage", "qualification", "rank_objective", "rotation", "lineup", "tactical_intent"]
TOPICS = ("qualification", "rank_objective", "rotation", "lineup", "tactical_intent")
VALUES: dict[str, frozenset[str]] = {
    "stage": frozenset({"group", "league", "knockout", "qualifying", "playoff", "friendly"}),
    "qualification": frozenset({"qualified", "not_yet_qualified", "eliminated"}),
    "rank_objective": frozenset({"draw_sufficient_for_first_place", "draw_sufficient_for_qualification",
                                "win_required_for_first_place", "win_required_for_qualification",
                                "position_locked", "conditional_on_other_fixture"}),
    "rotation": frozenset({"planned", "not_planned", "observed", "not_observed"}),
    "lineup": frozenset({"starting_xi"}),
    "tactical_intent": frozenset({"attack_oriented", "balanced", "defense_oriented"}),
}


def _time(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Competition context requires timezone-aware times")
    return parsed.astimezone(timezone.utc)


def _identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value)) and not re.match(r"^[A-Za-z]:", value)


class CompetitionObservation(Contract):
    """Owner record. `verified_by` names a trusted upstream verification step.

    A verifier name is provenance, not a credential: the owner is responsible for
    the actual check. LLM, analyst, and synthetic records cannot become verified
    facts merely by supplying the name of a verifier.
    """
    evidence_id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    competition_id: str = Field(min_length=1)
    team_id: str | None = None
    topic: Topic
    value: str
    available_at: AwareDatetime
    source_published_at: AwareDatetime | None = None
    observed_at: AwareDatetime | None = None
    source_id: str = Field(min_length=1)
    source_kind: Literal["official", "structured_provider", "reporter", "analyst", "llm", "synthetic"]
    epistemic_status: Literal["verified_fact", "hypothesis", "unverified", "conflicting"] = "unverified"
    observation_scope: Literal["pre_match", "post_match"] = "pre_match"
    availability_status: Literal["AVAILABLE", "UNAVAILABLE", "STALE", "FAILED"] = "AVAILABLE"
    verified_by: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    probability_basis: str | None = None
    premise_evidence_ids: list[str] = Field(default_factory=list)
    condition_fixture_ids: list[str] = Field(default_factory=list)
    objective_target: Literal["qualification", "first_place"] | None = None
    conditional_target: Literal["qualification", "first_place"] | None = None
    player_ids: list[str] = Field(default_factory=list)
    content_hash: str | None = None

    @model_validator(mode="after")
    def semantics(self):
        identifiers = [self.evidence_id, self.match_id, self.competition_id, self.source_id]
        identifiers += self.premise_evidence_ids + self.condition_fixture_ids + self.player_ids
        identifiers += [item for item in (self.team_id, self.verified_by, self.probability_basis,
                                         self.content_hash) if item is not None]
        if any(not _identifier(item) for item in identifiers):
            raise ValueError("Context references must be inert identifiers, not paths or free text")
        if self.value not in VALUES[self.topic]:
            raise ValueError(f"Unsupported value for context topic {self.topic}")
        if (self.topic == "stage") != (self.team_id is None):
            raise ValueError("Stage is match-scoped; other context observations require team_id")
        if self.epistemic_status == "verified_fact":
            if self.source_kind not in {"official", "structured_provider"} or not self.verified_by:
                raise ValueError("Verified facts require a verified official/provider record")
            if self.topic == "tactical_intent":
                raise ValueError("Pre-match intentions remain hypotheses")
            if self.probability is not None:
                raise ValueError("Fact verification and hypothesis probability are separate")
        if self.probability is not None:
            if self.epistemic_status != "hypothesis" or not self.probability_basis:
                raise ValueError("Probability requires hypothesis status and an explicit basis")
        if self.player_ids and self.topic != "lineup":
            raise ValueError("Only lineup evidence can carry player_ids")
        if self.topic == "lineup" and not self.player_ids:
            raise ValueError("A lineup claim requires player identities; otherwise leave it unknown")
        if len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("Lineup player identities must be unique")
        if self.evidence_id in self.premise_evidence_ids:
            raise ValueError("Evidence cannot be its own premise")
        if self.value == "conditional_on_other_fixture":
            if not self.condition_fixture_ids or not self.conditional_target or not self.premise_evidence_ids:
                raise ValueError("Conditional objectives require fixture references, target, and supporting premises")
            if self.match_id in self.condition_fixture_ids:
                raise ValueError("Other-fixture conditions must reference another fixture")
        elif self.condition_fixture_ids or self.conditional_target:
            raise ValueError("Only conditional objectives can carry other-fixture conditions")
        if self.topic != "rank_objective" and self.objective_target is not None:
            raise ValueError("Only rank objectives can carry objective_target")
        if self.topic == "rank_objective":
            implied = self.conditional_target if self.value == "conditional_on_other_fixture" else (
                "qualification" if self.value.endswith("_qualification") else (
                    "first_place" if self.value.endswith("_first_place") else None))
            if implied is None and self.objective_target is None:
                raise ValueError("Position-locked objectives require an explicit objective_target")
            if self.objective_target is not None and implied is not None and self.objective_target != implied:
                raise ValueError("Objective target disagrees with the stated rank condition")
            self.objective_target = self.objective_target or implied
        return self


class ContextClaim(Contract):
    """Fixed public slot: unknown has no substitute Boolean or numerical value."""
    status: Literal["verified_fact", "hypothesis", "unverified", "conflicting", "unknown"] = "unknown"
    value: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    probability_basis: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    # Summary only: independent dimensions are not contradictory alternatives.
    dimension_values: dict[str, str | None] = Field(default_factory=dict)


class CompetitionContext(Contract):
    schema_version: str = CONTEXT_VERSION
    match_id: str
    competition_id: str
    prediction_cutoff: AwareDatetime
    kickoff: AwareDatetime
    stage: ContextClaim = Field(default_factory=ContextClaim)
    teams: dict[str, dict[str, ContextClaim]]
    rank_objectives: dict[str, dict[str, ContextClaim]] = Field(default_factory=dict)
    rotation_states: dict[str, dict[str, ContextClaim]] = Field(default_factory=dict)
    evidence: dict[str, CompetitionObservation] = Field(default_factory=dict)
    readiness: Literal["no_context_observed", "synthetic_only", "partially_observed"]
    interpretation: str = "qualification_does_not_imply_rotation_or_weakness; incentives_do_not_identify_tactics"


def _claim(records: list[CompetitionObservation]) -> ContextClaim:
    if not records:
        return ContextClaim()
    values = sorted({record.value for record in records})
    evidence_ids = sorted(record.evidence_id for record in records)
    lineup_conflict = records[0].topic == "lineup" and len({tuple(sorted(record.player_ids)) for record in records}) > 1
    if len(values) != 1 or lineup_conflict or any(record.epistemic_status == "conflicting" for record in records):
        return ContextClaim(status="conflicting", evidence_ids=evidence_ids, alternatives=values)
    # Preserve unresolved epistemic status; hypotheses do not acquire the status
    # of an unrelated verified source merely because both contain the same word.
    statuses = {record.epistemic_status for record in records}
    status = "verified_fact" if statuses == {"verified_fact"} else (
        "hypothesis" if statuses <= {"verified_fact", "hypothesis"} else "unverified")
    probabilities = {(record.probability, record.probability_basis) for record in records}
    probability, basis = next(iter(probabilities)) if len(probabilities) == 1 else (None, None)
    return ContextClaim(status=status, value=values[0], evidence_ids=evidence_ids,
                        probability=probability, probability_basis=basis)


def _dimension(record: CompetitionObservation) -> str | None:
    if record.topic == "rank_objective":
        return record.objective_target
    if record.topic == "rotation":
        return "planned" if record.value in {"planned", "not_planned"} else "actual"
    return None


def _dimensional_claims(records: list[CompetitionObservation], dimensions: tuple[str, ...]) -> dict[str, ContextClaim]:
    return {dimension: _claim([record for record in records if _dimension(record) == dimension])
            for dimension in dimensions}


def _summary(claims: dict[str, ContextClaim]) -> ContextClaim:
    visible = {dimension: claim for dimension, claim in claims.items() if claim.status != "unknown"}
    if not visible:
        return ContextClaim()
    if len(visible) == 1:
        return next(iter(visible.values())).model_copy(deep=True)
    statuses = {claim.status for claim in visible.values()}
    status = "conflicting" if "conflicting" in statuses else (
        "verified_fact" if statuses == {"verified_fact"} else (
            "hypothesis" if statuses <= {"verified_fact", "hypothesis"} else "unverified"))
    return ContextClaim(status=status, value=None,
                        evidence_ids=sorted({ref for claim in visible.values() for ref in claim.evidence_ids}),
                        dimension_values={dimension: claim.value for dimension, claim in visible.items()})


def project_competition_context(
    *,
    fixture: Fixture | Mapping[str, Any],
    prediction_cutoff: datetime | str,
    observations: Iterable[Mapping[str, Any] | CompetitionObservation] = (),
    acquired_evidence_ids: Collection[str] = (),
) -> CompetitionContext:
    """Return a fixed, source/time/team-checked public projection.

    Unpurchased and unavailable payloads are dropped *before* schema validation,
    including malformed values. Purchased future/post-match/wrong-team evidence
    raises rather than silently becoming a usable observation. All hypothesis
    premises must themselves survive this projection. No rank rule is assumed:
    missing standings/tie-break verification leaves the corresponding slot unknown.
    """
    fixture = fixture if isinstance(fixture, Fixture) else Fixture.model_validate(fixture)
    cutoff = _time(prediction_cutoff)
    kickoff = _time(fixture.kickoff)
    if cutoff >= kickoff:
        raise ValueError("Competition context requires prediction_cutoff < kickoff")
    if fixture.home_team_id == fixture.away_team_id:
        raise ValueError("Fixture requires two different teams")
    if any(not _identifier(value) for value in (fixture.match_id, fixture.competition_id,
                                               fixture.home_team_id, fixture.away_team_id)):
        raise ValueError("Fixture references must be inert identifiers")
    acquired = set(acquired_evidence_ids)
    records: dict[str, CompetitionObservation] = {}
    for candidate in observations:
        raw = candidate.model_dump() if isinstance(candidate, CompetitionObservation) else candidate
        if not isinstance(raw, Mapping):
            raise ValueError("Context observations must be mappings")
        if raw.get("evidence_id") not in acquired:
            continue
        if raw.get("availability_status", "AVAILABLE") != "AVAILABLE":
            continue
        # Missing purchased premises keep a derived claim hidden; do not parse
        # its conclusion (which may include owner-side payload) prematurely.
        premises = raw.get("premise_evidence_ids", [])
        if not isinstance(premises, list) or any(not isinstance(item, str) for item in premises):
            raise ValueError("Premise evidence identifiers must be a list of strings")
        if not set(premises) <= acquired:
            continue
        record = CompetitionObservation.model_validate(raw)
        if record.match_id != fixture.match_id or record.competition_id != fixture.competition_id:
            raise ValueError("Competition evidence does not belong to the fixture")
        if record.team_id is not None and record.team_id not in {fixture.home_team_id, fixture.away_team_id}:
            raise ValueError("Competition evidence does not belong to a fixture team")
        if record.observation_scope != "pre_match":
            raise ValueError("Post-match observations are excluded from pre-match context")
        times = [record.available_at, record.source_published_at, record.observed_at]
        if any(_time(value) > cutoff for value in times if value is not None):
            raise ValueError("Competition evidence is newer than prediction_cutoff")
        if any(_time(value) > _time(record.available_at) for value in times[1:] if value is not None):
            raise ValueError("Evidence cannot be available before publication/observation")
        if record.evidence_id in records:
            raise ValueError("Duplicate competition evidence identity")
        records[record.evidence_id] = record
    # A purchased-but-missing or unavailable premise is not supporting evidence.
    # Resolve a dependency DAG to avoid cyclic hypotheses validating one another.
    accepted: dict[str, CompetitionObservation] = {}
    pending = dict(records)
    while pending:
        ready = [key for key, record in pending.items() if set(record.premise_evidence_ids) <= set(accepted)]
        if not ready:
            break
        for key in sorted(ready):
            record = pending.pop(key)
            if record.epistemic_status == "verified_fact" and any(
                    accepted[premise].epistemic_status != "verified_fact"
                    for premise in record.premise_evidence_ids):
                raise ValueError("Verified conclusions require verified supporting premises")
            accepted[key] = record
    for record in accepted.values():
        if record.epistemic_status != "verified_fact":
            continue
        for premise_id in record.premise_evidence_ids:
            premise = accepted[premise_id]
            peers = [other for other in accepted.values()
                     if other.topic == premise.topic and other.team_id == premise.team_id
                     and _dimension(other) == _dimension(premise)]
            if _claim(peers).status != "verified_fact":
                raise ValueError("Verified conclusions require unconflicted verified supporting premises")
    stage = _claim([record for record in accepted.values() if record.topic == "stage"])
    teams = {}
    rank_objectives = {}
    rotation_states = {}
    for team in (fixture.home_team_id, fixture.away_team_id):
        team_records = [record for record in accepted.values() if record.team_id == team]
        rank_objectives[team] = _dimensional_claims(
            [record for record in team_records if record.topic == "rank_objective"], ("qualification", "first_place"))
        rotation_states[team] = _dimensional_claims(
            [record for record in team_records if record.topic == "rotation"], ("planned", "actual"))
        teams[team] = {topic: _claim([record for record in team_records if record.topic == topic])
                       for topic in TOPICS if topic not in {"rank_objective", "rotation"}}
        teams[team]["rank_objective"] = _summary(rank_objectives[team])
        teams[team]["rotation"] = _summary(rotation_states[team])
    readiness = "no_context_observed" if not accepted else (
        "synthetic_only" if all(record.source_kind == "synthetic" for record in accepted.values())
        else "partially_observed")
    return CompetitionContext(match_id=fixture.match_id, competition_id=fixture.competition_id,
                              prediction_cutoff=cutoff, kickoff=kickoff, stage=stage, teams=teams,
                              rank_objectives=rank_objectives, rotation_states=rotation_states,
                              evidence={key: accepted[key] for key in sorted(accepted)}, readiness=readiness)
