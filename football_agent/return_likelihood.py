"""Point-in-time source-return likelihoods for the finite EVSI reference model.

The inquiry return logger records what an Agent actually received after it
selected a source.  This module gives that log a narrow statistical bridge to
``P(Z | theta, source, time)``.  The bridge is deliberately conservative:

* an observed operational return is not a latent-state label;
* a match result cannot be used as the pre-query state without leakage;
* censored, unselected, late and unavailable rows contribute no return count;
* adaptive Agent choices and reused dependence clusters do not become an IID
  likelihood merely because they are numerous;
* even a supported point estimate remains uncalibrated until an external
  calibration protocol is run.

The resulting matrix can be converted into ``ObservationTool`` and passed to
the exact finite EVSI solver.  It is an explicitly scoped estimate, never an
automatic claim about an open-world football source.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .decision_value import ObservationTool
from .inquiry_returns import OUTCOMES
from .return_dataset import VERSION as RETURN_DATASET_VERSION
from .types import utc


VERSION = "return-likelihood-bridge-v1"
OBSERVED = "OBSERVED"
CENSORED_AT_CUTOFF = "CENSORED_AT_CUTOFF"
NOT_SELECTED = "NOT_SELECTED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
ROW_STATUSES = frozenset({OBSERVED, CENSORED_AT_CUTOFF, NOT_SELECTED, SOURCE_UNAVAILABLE})
ORIGINS = frozenset({"constructed", "operational_network", "unknown"})
LABEL_BASES = frozenset({"constructed_pre_query_state", "pre_query_external_annotation", "match_result", None})
SUPPORTED_SELECTION_DESIGNS = frozenset({
    "constructed_randomized_source_probe",
    "pre_query_randomized_source_probe",
    "conditional_uniform_source_probe",
})
SUPPORTED_ESTIMATE_STATUSES = frozenset({
    "SUPPORTED_CONSTRUCTED_POINT_ESTIMATE",
    "SUPPORTED_PREQUERY_ANNOTATED_POINT_ESTIMATE",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _states(values: Sequence[str], name: str = "states") -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of unique strings")
    result = tuple(_text(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return result


@dataclass(frozen=True)
class ReturnObservation:
    """One normalized source return with its point-in-time boundary.

    ``latent_state`` is only an annotation available before the query.  It is
    intentionally separate from ``return_label``: labels such as
    ``source_support`` describe extraction, while ``theta`` describes a
    pre-query fixture or annotation used for conditional modeling.
    """

    row_id: str
    source_id: str
    time_bucket: str
    query_at: str
    prediction_cutoff: str
    status: str
    return_label: str | None
    available_at: str | None
    observed_at: str | None
    origin: str
    latent_state: str | None
    latent_state_basis: str | None
    selection_design: str
    cluster_id: str
    selection_probability: float | None = None
    selection_record_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _text(self.row_id, "row_id"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "time_bucket", _text(self.time_bucket, "time_bucket"))
        object.__setattr__(self, "query_at", utc(self.query_at).isoformat())
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())
        if self.status not in ROW_STATUSES:
            raise ValueError(f"Unsupported return row status: {self.status}")
        if self.origin not in ORIGINS:
            raise ValueError(f"Unsupported return evidence origin: {self.origin}")
        object.__setattr__(self, "return_label", _optional_text(self.return_label, "return_label"))
        if self.return_label is not None and self.return_label not in OUTCOMES:
            raise ValueError(f"return_label must be one of {OUTCOMES}")
        object.__setattr__(self, "available_at", None if self.available_at is None else utc(self.available_at).isoformat())
        object.__setattr__(self, "observed_at", None if self.observed_at is None else utc(self.observed_at).isoformat())
        object.__setattr__(self, "latent_state", _optional_text(self.latent_state, "latent_state"))
        if self.latent_state_basis not in LABEL_BASES:
            raise ValueError(f"Unsupported latent_state_basis: {self.latent_state_basis}")
        if self.latent_state is None and self.latent_state_basis is not None:
            raise ValueError("A latent_state_basis requires a latent_state")
        if self.latent_state is not None and self.latent_state_basis is None:
            raise ValueError("A latent_state requires an explicit pre-query label basis")
        object.__setattr__(self, "selection_design", _text(self.selection_design, "selection_design"))
        object.__setattr__(self, "cluster_id", _text(self.cluster_id, "cluster_id"))
        if self.selection_probability is not None:
            probability = _finite_nonnegative(self.selection_probability, "selection_probability")
            if probability > 1:
                raise ValueError("selection_probability must be at most one")
            object.__setattr__(self, "selection_probability", probability)
        if self.selection_record_id is not None:
            object.__setattr__(self, "selection_record_id", _text(self.selection_record_id, "selection_record_id"))
        if self.selection_probability is not None and self.selection_record_id is None:
            raise ValueError("selection_probability requires a selection_record_id")
        query, cutoff = utc(self.query_at), utc(self.prediction_cutoff)
        if query > cutoff:
            raise ValueError("query_at cannot be after prediction_cutoff")
        available = utc(self.available_at) if self.available_at else None
        observed = utc(self.observed_at) if self.observed_at else None
        if self.status == OBSERVED:
            if self.return_label is None or available is None or observed is None:
                raise ValueError("An observed row needs a label and both source/transport times")
            if available > cutoff or observed > cutoff:
                raise ValueError("Late source returns must be represented as censored rows")
            if observed < available:
                raise ValueError("observed_at cannot precede available_at")
        elif self.status in {CENSORED_AT_CUTOFF, NOT_SELECTED, SOURCE_UNAVAILABLE}:
            if self.return_label is not None:
                raise ValueError("Censored, unselected or unavailable rows have no observed return label")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReturnObservation":
        required = {
            "row_id", "source_id", "time_bucket", "query_at", "prediction_cutoff", "status",
            "return_label", "available_at", "observed_at", "origin", "latent_state",
            "latent_state_basis", "selection_design", "cluster_id",
        }
        if not isinstance(raw, Mapping) or not required <= set(raw):
            raise ValueError("A return observation is missing required point-in-time fields")
        values = {key: raw[key] for key in required}
        values["selection_probability"] = raw.get("selection_probability")
        values["selection_record_id"] = raw.get("selection_record_id")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id, "source_id": self.source_id, "time_bucket": self.time_bucket,
            "query_at": self.query_at, "prediction_cutoff": self.prediction_cutoff,
            "status": self.status, "return_label": self.return_label,
            "available_at": self.available_at, "observed_at": self.observed_at,
            "origin": self.origin, "latent_state": self.latent_state,
            "latent_state_basis": self.latent_state_basis,
            "selection_design": self.selection_design, "cluster_id": self.cluster_id,
            "selection_probability": self.selection_probability, "selection_record_id": self.selection_record_id,
        }


@dataclass(frozen=True)
class ReturnLikelihoodEstimate:
    """Auditable conditional likelihood output.

    ``likelihoods`` is ``None`` for every unsupported status.  A supported
    matrix is a Dirichlet-smoothed point estimate under an explicit sampling
    design; it carries no calibration interval or open-world generalization
    claim.
    """

    schema_version: str
    source_id: str
    time_bucket: str
    states: tuple[str, ...]
    outcomes: tuple[str, ...]
    alpha: float
    counts: dict[str, dict[str, int]]
    support_by_state: dict[str, int]
    likelihoods: dict[str, tuple[float, ...]] | None
    status: str
    evidence_class: str
    calibration_status: str
    selected_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    assumptions: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "source_id": self.source_id,
            "time_bucket": self.time_bucket, "states": list(self.states),
            "outcomes": list(self.outcomes), "alpha": self.alpha,
            "counts": {state: dict(values) for state, values in self.counts.items()},
            "support_by_state": dict(self.support_by_state),
            "likelihoods": None if self.likelihoods is None else {
                state: list(values) for state, values in self.likelihoods.items()
            },
            "status": self.status, "evidence_class": self.evidence_class,
            "calibration_status": self.calibration_status,
            "selected_row_ids": list(self.selected_row_ids),
            "excluded_row_ids": list(self.excluded_row_ids),
            "assumptions": list(self.assumptions),
        }

    def observation_tool(self, *, cost: float, description: str = "", evidence_id: str | None = None) -> ObservationTool:
        if self.status not in SUPPORTED_ESTIMATE_STATUSES or self.likelihoods is None:
            raise ValueError(f"A supported likelihood estimate is required, got {self.status}")
        return ObservationTool(self.outcomes, self.likelihoods, _finite_nonnegative(cost, "cost"), description, evidence_id)

    def query_value_assessment(self, *, candidate_id: str, model_row: Mapping[str, Any]):
        """Bridge a fitted return law to a scoped finite-model value row.

        A point estimate is sufficient for a declared finite-model calculation,
        but its ``NOT_CALIBRATED`` status remains visible in the returned
        assessment reason.  Unsupported or censored estimates stay unmeasured.
        """
        from .query_value_contract import QueryValueAssessment

        cost = model_row.get("cost") if isinstance(model_row, Mapping) else None
        if self.status not in SUPPORTED_ESTIMATE_STATUSES or self.likelihoods is None:
            return QueryValueAssessment.unmeasured(
                candidate_id, cost=cost,
                reason=f"return_likelihood_{self.status.lower()}",
            )
        if self.calibration_status != "NOT_CALIBRATED":
            raise ValueError("Only the explicit NOT_CALIBRATED point-estimate bridge is supported")
        if not isinstance(model_row, Mapping):
            raise ValueError("A finite model row mapping is required")
        if not model_row.get("feasible"):
            return QueryValueAssessment.unmeasured(
                candidate_id, cost=cost, reason="finite_model_row_unavailable",
            )
        if "immediate_evsi" not in model_row or "cost" not in model_row:
            raise ValueError("A feasible finite model row needs immediate_evsi and cost")
        return QueryValueAssessment.modeled(
            candidate_id, expected_value=model_row["immediate_evsi"], cost=model_row["cost"],
            reason="return_likelihood_point_estimate_NOT_CALIBRATED",
        )


def _empty_counts(states: Sequence[str], outcomes: Sequence[str]) -> dict[str, dict[str, int]]:
    return {state: {outcome: 0 for outcome in outcomes} for state in states}


class ReturnLikelihoodLedger:
    """Validated point-in-time rows and a deliberately narrow fitting gateway."""

    version = VERSION

    def __init__(self, rows: Iterable[ReturnObservation | Mapping[str, Any]]):
        normalized = tuple(row if isinstance(row, ReturnObservation) else ReturnObservation.from_mapping(row) for row in rows)
        if len({row.row_id for row in normalized}) != len(normalized):
            raise ValueError("Return observation row_id values must be unique")
        self.rows = normalized

    def summary(self) -> dict[str, Any]:
        statuses = Counter(row.status for row in self.rows)
        origins = Counter(row.origin for row in self.rows)
        labels = Counter(row.return_label for row in self.rows if row.return_label is not None)
        return {
            "schema_version": VERSION, "rows": len(self.rows),
            "status_counts": dict(statuses), "origin_counts": dict(origins),
            "return_label_counts": dict(labels),
            "sources": sorted({row.source_id for row in self.rows}),
            "time_buckets": sorted({row.time_bucket for row in self.rows}),
            "fitting_status": "DESCRIPTIVE_LOG_ONLY_UNTIL_EXPLICIT_STATE_AND_DESIGN",
            "calibrated": False,
        }

    def fit(
        self,
        *,
        source_id: str,
        time_bucket: str,
        states: Sequence[str],
        outcomes: Sequence[str] = OUTCOMES,
        alpha: float = 0.5,
        min_rows_per_state: int = 2,
        allow_repeated_clusters: bool = False,
        require_selection_records: bool = False,
    ) -> ReturnLikelihoodEstimate:
        """Fit a source/time-local Dirichlet point estimate when gates pass.

        The method never pools another source or time bucket.  Rows without a
        pre-query state, rows selected adaptively by the Agent, and rows that
        reuse a dependence cluster stop at an explicit diagnostic status.
        """
        source_id, time_bucket = _text(source_id, "source_id"), _text(time_bucket, "time_bucket")
        states = _states(states)
        outcomes = _states(outcomes, "outcomes")
        if any(outcome not in OUTCOMES for outcome in outcomes):
            raise ValueError(f"outcomes must be drawn from {OUTCOMES}")
        alpha = _finite_nonnegative(alpha, "alpha")
        if alpha <= 0:
            raise ValueError("alpha must be positive for Dirichlet smoothing")
        if type(min_rows_per_state) is not int or min_rows_per_state < 1:
            raise ValueError("min_rows_per_state must be a positive integer")
        matching = [row for row in self.rows if row.source_id == source_id and row.time_bucket == time_bucket]
        selected = [row for row in matching if row.status == OBSERVED and row.return_label in outcomes]
        excluded = [row for row in self.rows if row not in selected]
        counts = _empty_counts(states, outcomes)
        support = {state: 0 for state in states}

        def base(status: str, *, assumptions: Sequence[str] = ()) -> ReturnLikelihoodEstimate:
            return ReturnLikelihoodEstimate(
                VERSION, source_id, time_bucket, states, outcomes, alpha, counts, support, None,
                status, "DESCRIPTIVE_OBSERVATIONS_ONLY", "NOT_CALIBRATED",
                tuple(row.row_id for row in selected), tuple(row.row_id for row in excluded), tuple(assumptions),
            )

        if not matching:
            return base("NO_ROWS_FOR_SOURCE_TIME", assumptions=("No cross-source or cross-time pooling was performed.",))
        if any(row.origin == "unknown" for row in matching):
            return base("UNSUPPORTED_UNKNOWN_ORIGIN", assumptions=("Evidence origin must be explicit before fitting.",))
        if any(row.latent_state_basis == "match_result" for row in matching):
            return base("UNSUPPORTED_MATCH_RESULT_LEAKAGE", assumptions=("A post-event match result cannot define pre-query theta.",))
        if any(row.status == OBSERVED and row.return_label not in outcomes for row in matching):
            return base("UNSUPPORTED_OUTCOME_SCOPE", assumptions=("Requested outcome vocabulary does not cover every observed return.",))
        if any(row.selection_design not in SUPPORTED_SELECTION_DESIGNS for row in matching):
            return base("UNSUPPORTED_ADAPTIVE_SELECTION", assumptions=(
                "Agent-selected rows remain descriptive unless a source-probe design is logged.",
            ))
        if require_selection_records and any(
            row.selection_probability is None or row.selection_record_id is None for row in selected
        ):
            return base("UNSUPPORTED_MISSING_PROBE_RECORD", assumptions=(
                "Strict source-probe fitting requires a bound selection record and conditional probability.",
            ))
        if any(row.origin != matching[0].origin for row in matching):
            return base("UNSUPPORTED_MIXED_ORIGIN", assumptions=("Constructed and operational rows are never pooled.",))
        if not all(row.latent_state is not None for row in selected):
            return base("UNSUPPORTED_NO_LATENT_STATE", assumptions=(
                "A source return category is not a latent world-state label.",
            ))
        if any(row.latent_state not in states for row in selected):
            return base("UNSUPPORTED_STATE_SCOPE", assumptions=("Every theta label must be declared before fitting.",))
        if not allow_repeated_clusters and len({row.cluster_id for row in selected}) != len(selected):
            return base("UNSUPPORTED_DEPENDENCE_CLUSTER_REUSE", assumptions=(
                "Repeated pages or reports in one dependence cluster are not IID observations.",
            ))

        for row in selected:
            counts[row.latent_state][row.return_label] += 1
            support[row.latent_state] += 1
        if any(support[state] < min_rows_per_state for state in states):
            return base("INSUFFICIENT_SUPPORT", assumptions=(
                f"At least {min_rows_per_state} observed rows per theta state are required.",
            ))
        likelihoods = {}
        width = len(outcomes)
        for state in states:
            denominator = support[state] + alpha * width
            likelihoods[state] = tuple((counts[state][outcome] + alpha) / denominator for outcome in outcomes)
        status = (
            "SUPPORTED_CONSTRUCTED_POINT_ESTIMATE"
            if matching[0].origin == "constructed"
            else "SUPPORTED_PREQUERY_ANNOTATED_POINT_ESTIMATE"
        )
        return ReturnLikelihoodEstimate(
            VERSION, source_id, time_bucket, states, outcomes, alpha, counts, support, likelihoods,
            status, "DECLARED_POINT_ESTIMATE_NOT_CALIBRATED", "NOT_CALIBRATED",
            tuple(row.row_id for row in selected), tuple(row.row_id for row in excluded), (
                "Rows are source/time-local and use only cutoff-eligible observed returns.",
                "Dirichlet smoothing alpha is declared by the experiment, not learned from a match result.",
                "A supported point estimate is not an out-of-sample calibration result.",
            ),
        )


def from_return_dataset(
    dataset: Mapping[str, Any],
    *,
    source_id_resolver: Callable[[Mapping[str, Any]], str],
    return_label_resolver: Callable[[Mapping[str, Any]], str],
    latent_state_resolver: Callable[[Mapping[str, Any]], tuple[str | None, str | None]] | None = None,
    time_bucket_resolver: Callable[[Mapping[str, Any]], str] | None = None,
    selection_design_resolver: Callable[[Mapping[str, Any]], str] | None = None,
) -> ReturnLikelihoodLedger:
    """Adapt the audited return dataset without guessing missing labels.

    Resolvers are mandatory for source identity and return category.  This is
    intentional: a URL host, HTTP status or match result alone does not define
    a source-specific semantic return or a latent state.  The adapter keeps
    rows censored at the dataset ``as_of`` cutoff and leaves unobserved labels
    empty.
    """
    if not isinstance(dataset, Mapping) or dataset.get("schema_version") != RETURN_DATASET_VERSION:
        raise ValueError("Expected an inquiry return dataset v2")
    if not callable(source_id_resolver) or not callable(return_label_resolver):
        raise ValueError("Explicit source and return-label resolvers are required")
    as_of = utc(dataset["as_of"])
    rows = []
    for raw in dataset.get("rows", []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("pre_query"), Mapping):
            raise ValueError("Malformed return dataset row")
        pre, observed = raw["pre_query"], raw.get("observed_return")
        source_id = _text(source_id_resolver(raw), "resolved source_id")
        time_bucket = _text(time_bucket_resolver(raw) if time_bucket_resolver else "UNSCOPED", "resolved time_bucket")
        design = _text(selection_design_resolver(raw) if selection_design_resolver else "unlogged", "resolved selection_design")
        latent_state, latent_basis = (None, None)
        if latent_state_resolver:
            resolved = latent_state_resolver(raw)
            if not isinstance(resolved, tuple) or len(resolved) != 2:
                raise ValueError("latent_state_resolver must return (state, basis)")
            latent_state, latent_basis = resolved
        query_at = pre["query_decided_at"]
        if observed is None:
            rows.append(ReturnObservation(
                row_id=raw["row_id"], source_id=source_id, time_bucket=time_bucket,
                query_at=query_at, prediction_cutoff=as_of.isoformat(), status=CENSORED_AT_CUTOFF,
                return_label=None, available_at=None, observed_at=None, origin=raw["origin"],
                latent_state=latent_state, latent_state_basis=latent_basis,
                selection_design=design, cluster_id=raw["cluster_id"],
            ))
            continue
        available_at = observed.get("available_at")
        if available_at is None or utc(available_at) > as_of:
            rows.append(ReturnObservation(
                row_id=raw["row_id"], source_id=source_id, time_bucket=time_bucket,
                query_at=query_at, prediction_cutoff=as_of.isoformat(), status=CENSORED_AT_CUTOFF,
                return_label=None, available_at=available_at, observed_at=available_at, origin=raw["origin"],
                latent_state=latent_state, latent_state_basis=latent_basis,
                selection_design=design, cluster_id=raw["cluster_id"],
            ))
            continue
        label = _text(return_label_resolver(raw), "resolved return_label")
        rows.append(ReturnObservation(
            row_id=raw["row_id"], source_id=source_id, time_bucket=time_bucket,
            query_at=query_at, prediction_cutoff=as_of.isoformat(), status=OBSERVED,
            return_label=label, available_at=available_at, observed_at=available_at, origin=raw["origin"],
            latent_state=latent_state, latent_state_basis=latent_basis,
            selection_design=design, cluster_id=raw["cluster_id"],
        ))
    return ReturnLikelihoodLedger(rows)
