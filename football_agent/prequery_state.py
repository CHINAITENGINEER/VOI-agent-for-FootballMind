"""Point-in-time provenance for latent states used by source-probe calibration.

Qualification, expected rotation and tactical intent are separate pre-query
state labels.  This module records why a label exists and when its supporting
source was available; it never derives one state from another or accepts a
post-match result as a pre-query annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .types import utc


VERSION = "prequery-state-annotation-v1"
BASES = frozenset({"constructed_pre_query_state", "pre_query_external_annotation"})
PROVENANCE_KINDS = frozenset({
    "constructed_fixture", "official_pre_match_rule", "pre_match_statement",
    "pre_match_lineup", "pre_match_schedule", "external_pre_match_annotation",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class PreQueryStateAnnotation:
    annotation_id: str
    target_key: str
    state_variable: str
    value: str
    basis: str
    provenance_kind: str
    source_ids: tuple[str, ...]
    available_at: str
    observed_at: str
    prediction_cutoff: str
    scope: str
    visibility: str = "private_evaluator_only"

    def __post_init__(self):
        for field in ("annotation_id", "target_key", "state_variable", "value", "scope"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.basis not in BASES:
            raise ValueError("Unsupported pre-query annotation basis")
        if self.provenance_kind not in PROVENANCE_KINDS:
            raise ValueError("Unsupported pre-query annotation provenance kind")
        if self.basis == "constructed_pre_query_state" and self.provenance_kind != "constructed_fixture":
            raise ValueError("Constructed theta labels need constructed_fixture provenance")
        if self.basis == "pre_query_external_annotation" and self.provenance_kind == "constructed_fixture":
            raise ValueError("External theta labels need non-constructed provenance")
        source_ids = tuple(_text(source_id, "source_id") for source_id in self.source_ids)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Annotation source IDs must be unique")
        if self.basis == "pre_query_external_annotation" and not source_ids:
            raise ValueError("External theta labels need at least one source ID")
        object.__setattr__(self, "source_ids", source_ids)
        for field in ("available_at", "observed_at", "prediction_cutoff"):
            object.__setattr__(self, field, utc(getattr(self, field)).isoformat())
        available, observed, cutoff = utc(self.available_at), utc(self.observed_at), utc(self.prediction_cutoff)
        if available > observed or observed > cutoff:
            raise ValueError("Pre-query annotation times must satisfy available_at <= observed_at <= cutoff")
        if self.visibility != "private_evaluator_only":
            raise ValueError("State annotations must stay outside the Agent-visible candidate menu")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PreQueryStateAnnotation":
        fields = ("annotation_id", "target_key", "state_variable", "value", "basis",
                  "provenance_kind", "source_ids", "available_at", "observed_at",
                  "prediction_cutoff", "scope")
        if not isinstance(raw, Mapping) or any(field not in raw for field in fields):
            raise ValueError("Pre-query annotation is missing provenance or timing fields")
        return cls(**{field: raw[field] for field in fields}, visibility=raw.get("visibility", "private_evaluator_only"))

    def validate_for(self, *, target_key: str, prediction_cutoff: str) -> None:
        if self.target_key != _text(target_key, "target_key"):
            raise ValueError("Annotation target differs from the frozen probe target")
        if utc(prediction_cutoff).isoformat() != self.prediction_cutoff:
            raise ValueError("Annotation cutoff differs from the frozen probe cutoff")
        if utc(self.available_at) > utc(prediction_cutoff):
            raise ValueError("Annotation was not available at the prediction cutoff")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "annotation_id": self.annotation_id,
            "target_key": self.target_key, "state_variable": self.state_variable,
            "value": self.value, "basis": self.basis, "provenance_kind": self.provenance_kind,
            "source_ids": list(self.source_ids), "available_at": self.available_at,
            "observed_at": self.observed_at, "prediction_cutoff": self.prediction_cutoff,
            "scope": self.scope, "visibility": self.visibility,
        }


class PreQueryStateRegistry:
    """A pinned collection of independent state annotations."""

    version = VERSION

    def __init__(self, annotations: Iterable[PreQueryStateAnnotation | Mapping[str, Any]]):
        normalized = tuple(annotation if isinstance(annotation, PreQueryStateAnnotation)
                          else PreQueryStateAnnotation.from_mapping(annotation) for annotation in annotations)
        if len({annotation.annotation_id for annotation in normalized}) != len(normalized):
            raise ValueError("Annotation IDs must be unique")
        self.annotations = normalized

    def eligible(self, *, target_key: str, prediction_cutoff: str) -> tuple[PreQueryStateAnnotation, ...]:
        result = []
        for annotation in self.annotations:
            try:
                annotation.validate_for(target_key=target_key, prediction_cutoff=prediction_cutoff)
            except ValueError:
                continue
            result.append(annotation)
        return tuple(result)

    def require(self, annotation_id: str) -> PreQueryStateAnnotation:
        for annotation in self.annotations:
            if annotation.annotation_id == annotation_id:
                return annotation
        raise ValueError("Unknown pre-query annotation")

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": VERSION, "annotations": [annotation.manifest() for annotation in self.annotations],
                "state_variables_are_independent": True,
                "post_match_result_derivation": False}
