"""Append-only source-backed context registration and explicit as-of projection.

Raw bytes use the existing RawArchive. Registrations add an extraction anchor and
actual operational availability; the adapter never promotes a mapped claim into
a verified fact. Retrospective publication assumptions stay separate from actual
acquisition times, and post-match documents are evaluation-only.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import AwareDatetime, Field

from football_spider.models.contracts import Contract, Observation
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from .competition_context import CompetitionContext, CompetitionObservation, _identifier, _time, project_competition_context
from .contracts import Fixture


class SourceAnchor(Contract):
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceRegistration(Contract):
    schema_version: str = "competition-evidence-registration-v1"
    evidence_id: str
    fixture: Fixture
    raw_observation: Observation
    registered_at: AwareDatetime
    operational_available_at: AwareDatetime
    source_published_at: AwareDatetime | None = None
    publication_time_basis: Literal["explicit_timestamp_in_source", "unknown"] = "unknown"
    publication_time_anchor: SourceAnchor | None = None
    document_scope: Literal["pre_match_preview", "post_match_report", "unknown"]
    partition: Literal["candidate-evidence", "evaluation-quarantine"]
    claim: CompetitionObservation | None = None
    anchor: SourceAnchor | None = None
    extraction_verification: Literal["anchor_integrity_only_semantics_unverified", "no_claim"]


class ReconstructionAssumption(Contract):
    assumption_id: str
    assumed_available_at: AwareDatetime
    basis: Literal["source_publication_assumed_unchanged_since_publication"]


class RegistryProjection(Contract):
    schema_version: str = "competition-registry-projection-v1"
    information_mode: Literal["operational", "historical_reconstruction"]
    context: CompetitionContext
    exclusions: dict[str, str] = Field(default_factory=dict)
    actual_observation_times: dict[str, dict[str, str]] = Field(default_factory=dict)
    availability_assumptions: dict[str, ReconstructionAssumption] = Field(default_factory=dict)
    evidence_use: str = "source-backed_unverified_claims; not_policy_gain_ground_truth"


class CompetitionEvidenceRegistry:
    def __init__(self, root: Path, archive: RawArchive, *, clock=lambda: datetime.now(timezone.utc)):
        self.root, self.archive, self.clock = Path(root).resolve(), archive, clock

    def _record_path(self, partition: str, evidence_id: str) -> Path:
        if partition not in {"candidate-evidence", "evaluation-quarantine"}:
            raise ValueError("Unknown evidence partition")
        # Inert logical IDs may contain colons or Windows reserved names. They
        # are never filesystem names (in particular never NTFS stream syntax).
        filename = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest() + ".json"
        return self.root / partition / filename

    def _verified_raw_body(self, observation: Observation) -> bytes:
        body = self.archive.read_body(observation)
        payload = (self.archive.root / observation.payload_path).resolve()
        if not payload.name.endswith(".body.gz"):
            raise ValueError("Expected a committed RawArchive payload")
        metadata = payload.with_name(payload.name[:-len(".body.gz")] + ".observation.json")
        committed = Observation.model_validate_json(metadata.read_bytes())
        if committed.model_dump() != observation.model_dump():
            raise ValueError("Observation metadata differs from its immutable archive commit")
        return body

    def register(self, *, evidence_id: str, fixture: Fixture | Mapping[str, Any], observation: Observation,
                 document_scope: str, source_kind: str = "official", claim: Mapping[str, Any] | None = None,
                 anchor: SourceAnchor | Mapping[str, Any] | None = None,
                 source_published_at: datetime | str | None = None,
                 publication_time_anchor: SourceAnchor | Mapping[str, Any] | None = None,
                 publication_time_basis: str = "unknown") -> EvidenceRegistration:
        """Commit a new record; an existing evidence identity is never overwritten.

        The caller supplies an extraction and its exact raw-byte anchor. This
        proves the anchor is archived, not that its mapped meaning is correct.
        Only unverified claims/hypotheses are admitted at this boundary.
        """
        if not isinstance(evidence_id, str) or not _identifier(evidence_id):
            raise ValueError("Evidence identity must be an inert identifier")
        fixture = fixture if isinstance(fixture, Fixture) else Fixture.model_validate(fixture)
        now = _time(self.clock())
        if _time(observation.observed_at) > now:
            raise ValueError("Observation cannot be in the future of registration")
        body = self._verified_raw_body(observation)
        if observation.http_status is None or not 200 <= observation.http_status < 300 or observation.error_class:
            raise ValueError("Context registration requires a successful archived response")
        published = _time(source_published_at) if source_published_at is not None else None
        if (published is None) != (publication_time_basis == "unknown"):
            raise ValueError("Publication basis and explicit source timestamp must agree")
        if published is not None and published > _time(observation.observed_at):
            raise ValueError("Source publication time exceeds actual observation time")
        parsed_publication_anchor = SourceAnchor.model_validate(publication_time_anchor) if publication_time_anchor is not None else None
        if published is not None:
            if parsed_publication_anchor is None:
                raise ValueError("Explicit publication time requires its raw timestamp anchor")
            start, end = parsed_publication_anchor.byte_start, parsed_publication_anchor.byte_end
            timestamp_bytes = body[start:end]
            if end <= start or end > len(body) or hashlib.sha256(timestamp_bytes).hexdigest() != parsed_publication_anchor.sha256:
                raise ValueError("Publication timestamp anchor checksum or range mismatch")
            if _time(timestamp_bytes.decode("utf-8").strip()) != published:
                raise ValueError("Publication timestamp differs from the anchored source timestamp")
        elif parsed_publication_anchor is not None:
            raise ValueError("Unknown publication time cannot carry a timestamp anchor")
        if document_scope != "pre_match_preview" and claim is not None:
            raise ValueError("Only pre-match previews may carry context candidate claims")
        parsed_anchor = SourceAnchor.model_validate(anchor) if anchor is not None else None
        if parsed_anchor is not None:
            if parsed_anchor.byte_end <= parsed_anchor.byte_start or parsed_anchor.byte_end > len(body):
                raise ValueError("Extraction anchor is outside the archived response")
            if hashlib.sha256(body[parsed_anchor.byte_start:parsed_anchor.byte_end]).hexdigest() != parsed_anchor.sha256:
                raise ValueError("Extraction anchor checksum mismatch")
        parsed_claim = None
        if claim is not None:
            if parsed_anchor is None:
                raise ValueError("Context candidates require an archived source anchor")
            allowed = {"topic", "value", "team_id", "epistemic_status", "probability", "probability_basis",
                       "premise_evidence_ids", "condition_fixture_ids", "objective_target", "conditional_target", "player_ids"}
            if set(claim) - allowed:
                raise ValueError("Claim input contains unsupported or owner-controlled fields")
            status = claim.get("epistemic_status", "unverified")
            if status not in {"unverified", "hypothesis"}:
                raise ValueError("Registration does not establish verified semantic facts")
            parsed_claim = CompetitionObservation.model_validate({
                **dict(claim), "epistemic_status": status, "evidence_id": evidence_id,
                "match_id": fixture.match_id, "competition_id": fixture.competition_id,
                "available_at": now, "observed_at": observation.observed_at,
                "source_published_at": published, "source_id": observation.source,
                "source_kind": source_kind, "observation_scope": "pre_match",
                "content_hash": observation.content_hash,
            })
            if parsed_claim.team_id is not None and parsed_claim.team_id not in {fixture.home_team_id, fixture.away_team_id}:
                raise ValueError("Candidate team does not belong to the fixture")
        partition = "candidate-evidence" if document_scope == "pre_match_preview" else "evaluation-quarantine"
        registration = EvidenceRegistration(
            evidence_id=evidence_id, fixture=fixture, raw_observation=observation, registered_at=now,
            operational_available_at=now, source_published_at=published, publication_time_basis=publication_time_basis,
            publication_time_anchor=parsed_publication_anchor,
            document_scope=document_scope, partition=partition, claim=parsed_claim, anchor=parsed_anchor,
            extraction_verification="anchor_integrity_only_semantics_unverified" if parsed_claim else "no_claim")
        content = registration.model_dump(mode="json")
        envelope = {"record": content, "record_sha256": hashlib.sha256(json_bytes(content)).hexdigest()}
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / "registration.lock")):
            if any(self._record_path(name, evidence_id).exists()
                   for name in ("candidate-evidence", "evaluation-quarantine")):
                raise ValueError("Append-only evidence identity already exists; use a new version identity")
            atomic_write(self._record_path(partition, evidence_id), json_bytes(envelope))
        return registration

    def read(self, evidence_id: str) -> EvidenceRegistration:
        if not isinstance(evidence_id, str) or not _identifier(evidence_id):
            raise ValueError("Evidence identity must be an inert identifier")
        paths = [self._record_path(name, evidence_id) for name in ("candidate-evidence", "evaluation-quarantine")]
        existing = [path for path in paths if path.exists()]
        if len(existing) != 1:
            raise ValueError("Evidence registration missing or ambiguous")
        envelope = json.loads(existing[0].read_bytes())
        if hashlib.sha256(json_bytes(envelope["record"])).hexdigest() != envelope["record_sha256"]:
            raise ValueError("Evidence registration checksum mismatch")
        record = EvidenceRegistration.model_validate(envelope["record"])
        if record.evidence_id != evidence_id or record.partition != existing[0].parent.name:
            raise ValueError("Evidence registration identity or partition mismatch")
        body = self._verified_raw_body(record.raw_observation)
        if record.anchor and hashlib.sha256(body[record.anchor.byte_start:record.anchor.byte_end]).hexdigest() != record.anchor.sha256:
            raise ValueError("Evidence anchor checksum mismatch")
        if record.operational_available_at < max(record.registered_at, record.raw_observation.observed_at):
            raise ValueError("Operational availability precedes actual observation/registration")
        return record

    def project(self, *, fixture: Fixture | Mapping[str, Any], prediction_cutoff: datetime | str,
                acquired_evidence_ids: Collection[str], information_mode: str = "operational",
                reconstruction_assumptions: Mapping[str, ReconstructionAssumption | Mapping[str, Any]] | None = None) -> RegistryProjection:
        if information_mode not in {"operational", "historical_reconstruction"}:
            raise ValueError("Unknown context information mode")
        if information_mode == "operational" and reconstruction_assumptions:
            raise ValueError("Operational projection does not accept historical availability assumptions")
        fixture = fixture if isinstance(fixture, Fixture) else Fixture.model_validate(fixture)
        cutoff = _time(prediction_cutoff)
        visible, exclusions, times, assumptions = [], {}, {}, {}
        for evidence_id in sorted(set(acquired_evidence_ids)):
            record = self.read(evidence_id)
            if record.fixture != fixture:
                raise ValueError("Registered context fixture differs from the requested fixture")
            times[evidence_id] = {
                "observed_at": record.raw_observation.observed_at.isoformat(),
                "registered_at": record.registered_at.isoformat(),
                "operational_available_at": record.operational_available_at.isoformat(),
            }
            if record.partition == "evaluation-quarantine":
                exclusions[evidence_id] = "evaluation_only_document"
                continue
            if record.claim is None:
                exclusions[evidence_id] = "no_structured_context_claim"
                continue
            available = record.operational_available_at
            observed = record.raw_observation.observed_at
            if information_mode == "historical_reconstruction":
                raw_assumption = (reconstruction_assumptions or {}).get(evidence_id)
                if raw_assumption is None:
                    exclusions[evidence_id] = "explicit_reconstruction_assumption_required"
                    continue
                assumption = ReconstructionAssumption.model_validate(raw_assumption)
                if not _identifier(assumption.assumption_id):
                    raise ValueError("Reconstruction assumption requires an inert identifier")
                if record.source_published_at is None:
                    exclusions[evidence_id] = "source_publication_timestamp_unknown"
                    continue
                if assumption.assumed_available_at < record.source_published_at:
                    raise ValueError("Assumed availability cannot precede source publication")
                available, observed = assumption.assumed_available_at, None
                assumptions[evidence_id] = assumption
            if available > cutoff:
                exclusions[evidence_id] = "not_available_at_prediction_cutoff"
                continue
            payload = record.claim.model_dump()
            payload.update(available_at=available, observed_at=observed)
            visible.append(payload)
        context = project_competition_context(fixture=fixture, prediction_cutoff=cutoff,
                                             observations=visible, acquired_evidence_ids=acquired_evidence_ids)
        return RegistryProjection(information_mode=information_mode, context=context, exclusions=exclusions,
                                  actual_observation_times=times, availability_assumptions=assumptions)
