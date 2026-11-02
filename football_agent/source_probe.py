"""A separate randomized source-probe arm for identifying return laws.

The FootballMind Agent remains adaptive in its decision arm.  This module is a
different, explicitly labeled calibration arm: before any source response is
seen, it samples one source uniformly from a common candidate set.  Its rows
can support ``P(Z | theta, source, time)`` after point-in-time and dependence
checks.  Probe records never overwrite or relabel adaptive Agent decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from .prequery_state import PreQueryStateAnnotation
from .return_likelihood import ReturnLikelihoodLedger, ReturnObservation
from .types import utc


VERSION = "randomized-source-probe-v1"
DESIGN = "conditional_uniform_source_probe"
STATE_BASES = frozenset({"constructed_pre_query_state", "pre_query_external_annotation"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("A nonnegative private integer seed is required")
    return value


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProbeCandidate:
    candidate_id: str
    source_id: str
    endpoint: str
    target_key: str
    time_bucket: str
    prediction_cutoff: str
    information_role: str
    document_scope: str

    def __post_init__(self):
        for field in ("candidate_id", "source_id", "endpoint", "target_key", "time_bucket",
                      "information_role", "document_scope"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProbeCandidate":
        fields = ("candidate_id", "source_id", "endpoint", "target_key", "time_bucket",
                  "prediction_cutoff", "information_role", "document_scope")
        if not isinstance(raw, Mapping) or any(field not in raw for field in fields):
            raise ValueError("A probe candidate is missing its fixed source and timing fields")
        return cls(**{field: raw[field] for field in fields})

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in (
            "candidate_id", "source_id", "endpoint", "target_key", "time_bucket",
            "prediction_cutoff", "information_role", "document_scope")}


@dataclass(frozen=True)
class ProbeSelection:
    selection_id: str
    task_id: str
    query_at: str
    target_key: str
    time_bucket: str
    prediction_cutoff: str
    latent_state: str
    latent_state_basis: str
    candidates: tuple[ProbeCandidate, ...]
    selected_candidate_id: str
    probabilities: dict[str, float]
    selected_probability: float
    seed_commitment: str
    selection_design: str = DESIGN
    latent_state_visibility: str = "private_evaluator_only"
    state_annotation_id: str | None = None
    state_annotation_variable: str | None = None
    state_annotation_available_at: str | None = None
    state_annotation_source_ids: tuple[str, ...] = ()
    state_annotation_provenance_kind: str | None = None
    state_annotation_scope: str | None = None

    def __post_init__(self):
        for field in ("selection_id", "task_id", "target_key", "time_bucket", "latent_state", "seed_commitment"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "query_at", utc(self.query_at).isoformat())
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())
        if self.latent_state_basis not in STATE_BASES:
            raise ValueError("A probe state needs an explicit pre-query label basis")
        if self.selection_design != DESIGN:
            raise ValueError("Unexpected source probe selection design")
        if self.latent_state_visibility != "private_evaluator_only":
            raise ValueError("Probe theta labels must stay outside the Agent-visible candidate menu")
        if self.state_annotation_id is None:
            if any(value is not None for value in (
                self.state_annotation_variable, self.state_annotation_available_at,
                self.state_annotation_provenance_kind, self.state_annotation_scope,
            )) or self.state_annotation_source_ids:
                raise ValueError("State annotation metadata requires an annotation ID")
            annotation_source_ids = ()
        else:
            object.__setattr__(self, "state_annotation_id", _text(self.state_annotation_id, "state_annotation_id"))
            for field in ("state_annotation_variable", "state_annotation_provenance_kind", "state_annotation_scope"):
                value = getattr(self, field)
                if value is None:
                    raise ValueError(f"{field} is required with a state annotation ID")
                object.__setattr__(self, field, _text(value, field))
            if self.state_annotation_available_at is None:
                raise ValueError("state_annotation_available_at is required with a state annotation ID")
            object.__setattr__(self, "state_annotation_available_at", utc(self.state_annotation_available_at).isoformat())
            annotation_source_ids = tuple(_text(value, "state_annotation_source_id") for value in self.state_annotation_source_ids)
            if self.state_annotation_provenance_kind != "constructed_fixture" and not annotation_source_ids:
                raise ValueError("A bound external state annotation needs source IDs")
            if len(set(annotation_source_ids)) != len(annotation_source_ids):
                raise ValueError("State annotation source IDs must be unique")
        object.__setattr__(self, "state_annotation_source_ids", annotation_source_ids)
        candidates = tuple(self.candidates)
        if len(candidates) < 2:
            raise ValueError("A randomized source probe needs at least two candidates")
        if any(not isinstance(candidate, ProbeCandidate) for candidate in candidates):
            raise ValueError("Probe candidates must be ProbeCandidate objects")
        if tuple(sorted(candidate.candidate_id for candidate in candidates)) != tuple(candidate.candidate_id for candidate in candidates):
            raise ValueError("Probe candidates must be stored in canonical candidate_id order")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("Probe candidate IDs must be unique")
        if len({candidate.source_id for candidate in candidates}) != len(candidates):
            raise ValueError("A probe candidate set needs distinct source IDs")
        if any(candidate.target_key != self.target_key or candidate.time_bucket != self.time_bucket
               or candidate.prediction_cutoff != self.prediction_cutoff for candidate in candidates):
            raise ValueError("All candidates must share target, time bucket and cutoff")
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if self.selected_candidate_id not in candidate_ids:
            raise ValueError("The selected candidate is outside the frozen candidate set")
        expected = 1 / len(candidates)
        if set(self.probabilities) != set(candidate_ids) or any(
            type(value) not in {int, float} or not math.isfinite(value) or abs(value - expected) > 1e-12
            for value in self.probabilities.values()
        ) or abs(math.fsum(self.probabilities.values()) - 1) > 1e-12:
            raise ValueError("Probe probabilities must be uniform conditional on the frozen candidate set")
        if abs(self.selected_probability - expected) > 1e-12:
            raise ValueError("Selected probability differs from the declared conditional design")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "selection_id": self.selection_id, "task_id": self.task_id,
            "query_at": self.query_at, "target_key": self.target_key, "time_bucket": self.time_bucket,
            "prediction_cutoff": self.prediction_cutoff, "latent_state": self.latent_state,
            "latent_state_basis": self.latent_state_basis,
            "latent_state_visibility": self.latent_state_visibility,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "probabilities": dict(self.probabilities), "selected_probability": self.selected_probability,
            "seed_commitment": self.seed_commitment, "selection_design": self.selection_design,
            "state_annotation": None if self.state_annotation_id is None else {
                "annotation_id": self.state_annotation_id,
                "state_variable": self.state_annotation_variable,
                "available_at": self.state_annotation_available_at,
                "source_ids": list(self.state_annotation_source_ids),
                "provenance_kind": self.state_annotation_provenance_kind,
                "scope": self.state_annotation_scope,
            },
        }


class SourceProbeDesign:
    """Construct and verify a pre-response uniform source-probe assignment."""

    version = VERSION

    @staticmethod
    def select(
        *,
        selection_id: str,
        task_id: str,
        query_at: str,
        candidates: Sequence[ProbeCandidate | Mapping[str, Any]],
        latent_state: str,
        latent_state_basis: str,
        seed: int,
        state_annotation: PreQueryStateAnnotation | None = None,
    ) -> ProbeSelection:
        selection_id, task_id = _text(selection_id, "selection_id"), _text(task_id, "task_id")
        seed = _seed(seed)
        normalized = tuple(sorted(
            (candidate if isinstance(candidate, ProbeCandidate) else ProbeCandidate.from_mapping(candidate)
             for candidate in candidates), key=lambda candidate: candidate.candidate_id))
        if not normalized:
            raise ValueError("At least two source candidates are required")
        target_key, time_bucket, cutoff = normalized[0].target_key, normalized[0].time_bucket, normalized[0].prediction_cutoff
        if any(candidate.target_key != target_key or candidate.time_bucket != time_bucket
               or candidate.prediction_cutoff != cutoff for candidate in normalized):
            raise ValueError("Candidate menu must be fixed to one target and prediction cutoff")
        annotation_fields: dict[str, Any] = {
            "state_annotation_id": None,
            "state_annotation_variable": None,
            "state_annotation_available_at": None,
            "state_annotation_source_ids": (),
            "state_annotation_provenance_kind": None,
            "state_annotation_scope": None,
        }
        if state_annotation is not None:
            if not isinstance(state_annotation, PreQueryStateAnnotation):
                raise ValueError("state_annotation must be a PreQueryStateAnnotation")
            state_annotation.validate_for(target_key=target_key, prediction_cutoff=cutoff)
            if latent_state != state_annotation.value:
                raise ValueError("Probe latent_state differs from the pinned state annotation")
            if latent_state_basis != state_annotation.basis:
                raise ValueError("Probe latent_state_basis differs from the pinned state annotation")
            annotation_fields = {
                "state_annotation_id": state_annotation.annotation_id,
                "state_annotation_variable": state_annotation.state_variable,
                "state_annotation_available_at": state_annotation.available_at,
                "state_annotation_source_ids": state_annotation.source_ids,
                "state_annotation_provenance_kind": state_annotation.provenance_kind,
                "state_annotation_scope": state_annotation.scope,
            }
        private_material = "|".join((selection_id, str(seed), *(candidate.candidate_id for candidate in normalized)))
        index = int(_hash(private_material), 16) % len(normalized)
        ids = tuple(candidate.candidate_id for candidate in normalized)
        probabilities = {candidate_id: 1 / len(ids) for candidate_id in ids}
        return ProbeSelection(
            selection_id=selection_id, task_id=task_id, query_at=query_at,
            target_key=target_key, time_bucket=time_bucket, prediction_cutoff=cutoff,
            latent_state=latent_state, latent_state_basis=latent_state_basis, candidates=normalized,
            selected_candidate_id=ids[index], probabilities=probabilities,
            selected_probability=probabilities[ids[index]],
            seed_commitment=_hash(f"{selection_id}|{seed}"),
            **annotation_fields,
        )

    @staticmethod
    def verify(selection: ProbeSelection, *, seed: int) -> None:
        if not isinstance(selection, ProbeSelection):
            raise ValueError("A ProbeSelection is required")
        seed = _seed(seed)
        if selection.seed_commitment != _hash(f"{selection.selection_id}|{seed}"):
            raise ValueError("Probe seed commitment differs")
        private_material = "|".join((selection.selection_id, str(seed), *(candidate.candidate_id for candidate in selection.candidates)))
        expected = selection.candidates[int(_hash(private_material), 16) % len(selection.candidates)].candidate_id
        if selection.selected_candidate_id != expected:
            raise ValueError("Probe selection differs from the committed uniform assignment")

    @staticmethod
    def bind_return(
        selection: ProbeSelection,
        *,
        seed: int,
        row_id: str,
        return_label: str,
        available_at: str,
        observed_at: str,
        cluster_id: str,
        origin: str = "constructed",
    ) -> ReturnObservation:
        SourceProbeDesign.verify(selection, seed=seed)
        candidate = next(item for item in selection.candidates if item.candidate_id == selection.selected_candidate_id)
        return ReturnObservation(
            row_id=row_id, source_id=candidate.source_id, time_bucket=selection.time_bucket,
            query_at=selection.query_at, prediction_cutoff=selection.prediction_cutoff,
            status="OBSERVED", return_label=return_label,
            available_at=available_at, observed_at=observed_at, origin=origin,
            latent_state=selection.latent_state, latent_state_basis=selection.latent_state_basis,
            selection_design=DESIGN, cluster_id=cluster_id,
            selection_probability=selection.selected_probability, selection_record_id=selection.selection_id,
        )

    @staticmethod
    def validate_bound_rows(selections: Iterable[ProbeSelection], rows: Iterable[ReturnObservation]) -> dict[str, Any]:
        selection_map = {selection.selection_id: selection for selection in selections}
        if len(selection_map) == 0:
            raise ValueError("At least one probe selection is required")
        seen = set()
        for row in rows:
            if not isinstance(row, ReturnObservation):
                raise ValueError("Bound probe rows must be ReturnObservation objects")
            if row.selection_record_id not in selection_map:
                raise ValueError("Return row references an unknown probe selection")
            if row.selection_record_id in seen:
                raise ValueError("A probe selection may have one observed return row")
            selection = selection_map[row.selection_record_id]
            candidate = next(item for item in selection.candidates if item.candidate_id == selection.selected_candidate_id)
            if row.source_id != candidate.source_id or row.selection_probability != selection.selected_probability:
                raise ValueError("Return row differs from its committed selected source or probability")
            if row.latent_state != selection.latent_state or row.latent_state_basis != selection.latent_state_basis:
                raise ValueError("Return row theta differs from the private probe annotation")
            seen.add(row.selection_record_id)
        return {
            "schema_version": VERSION, "status": "BOUND",
            "selection_count": len(selection_map), "observed_return_count": len(seen),
            "unobserved_selection_count": len(selection_map) - len(seen),
            "selection_design": DESIGN,
            "labels_visible_to_agent": False,
        }

    @staticmethod
    def ledger(selections: Iterable[ProbeSelection], rows: Iterable[ReturnObservation]) -> ReturnLikelihoodLedger:
        selections, rows = tuple(selections), tuple(rows)
        SourceProbeDesign.validate_bound_rows(selections, rows)
        return ReturnLikelihoodLedger(rows)
