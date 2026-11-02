"""Build the explicit pre-query feature vector for the synthetic g prior.

The synthetic teacher is deliberately not allowed to guess semantics from an
action id, URL, title, or a document returned after a query.  This module is
the narrow boundary where a candidate generator (or its caller) can attach a
versioned, pre-query declaration.  It validates the declaration and returns a
copy of the menu, leaving the original generator output untouched.

``build_prequery_prior_features`` is the optional deterministic builder at
that boundary.  It computes state geometry from the frozen public prediction
and accepts source semantics only from a caller-supplied capability registry;
it does not inspect candidate text or response bodies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any


SCHEMA = "footballmind-g-prior-features-v1"
PREQUERY_BUILDER_SCHEMA = "footballmind-g-prequery-feature-builder-v1"
# v1 remains available for replay compatibility.  New online pre-query
# callers can opt into this stricter contract: it requires canonical public
# state locations, normalized probabilities, and a genuinely pre-kickoff
# prediction cutoff instead of using legacy fallbacks.
STRICT_PREQUERY_BUILDER_SCHEMA = "footballmind-g-prequery-feature-builder-v2"
FIELD = "synthetic_prior_features"
ALIAS = "g_prior_features"
TEACHING_SCHEMA = "footballmind-g-teaching-features-v1"
INFORMATION_KINDS = (
    "lineup_injury",
    "tactical_plan",
    "motivation_rotation",
    "odds_shock",
    "weather",
    "historical_h2h",
)
CONTINUOUS_FIELDS = (
    "uncertainty",
    "decision_sensitivity",
    "source_reliability",
    "specificity",
    "redundancy",
    "misdirection_risk",
    "query_cost",
)
REQUIRED_FIELDS = ("information_kind", *CONTINUOUS_FIELDS)

# The newer ``synthetic-g-teaching-v1`` artifact has a different feature
# contract from the phase-4 bridge prior above.  Keep it under the same
# decision-local field, but require an explicit schema marker so a coefficient
# vector can never consume a feature from the other artifact by accident.
TEACHING_ACTIONS = ("lineup", "injury", "motivation", "weather")
TEACHING_SCENARIOS = (
    "rotation_sensitive", "tactical_sensitive", "stable", "weather_sensitive",
)
TEACHING_REQUIRED_FIELDS = (
    "schema", "action", "scenario", "entropy", "max_probability",
    "time_to_kickoff_hours", "source_reliability", "event_relevance",
    "tactical_sensitivity", "cost_units",
)

# These are deliberately separate from the prior vector itself.  A candidate
# producer may declare what a source can observe and what it costs, while the
# builder obtains state-dependent quantities (entropy, probability margin and
# time to kickoff) from the public pre-query snapshot.  It never inspects an
# action's URL, title, query text, or a returned document.
LEGACY_CAPABILITY_FIELDS = (
    "information_kind", "decision_sensitivity_multiplier",
    "source_reliability", "specificity", "redundancy",
    "misdirection_risk", "query_cost",
)
TEACHING_CAPABILITY_FIELDS = (
    "action", "scenario", "source_reliability", "event_relevance",
    "tactical_sensitivity", "cost_units",
)


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def validate_prior_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete, explicit pre-query feature declaration.

    No value is inferred from a candidate's label or source metadata.  Unknown
    fields are rejected so a new feature cannot silently acquire a coefficient
    from an older synthetic model artifact.
    """
    if not isinstance(features, Mapping):
        raise ValueError("prior features must be a mapping")
    keys = set(features)
    missing = set(REQUIRED_FIELDS) - keys
    extra = keys - set(REQUIRED_FIELDS)
    if missing:
        raise ValueError(f"prior features missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"prior features contain unknown fields: {sorted(extra)}")
    kind = features["information_kind"]
    if not isinstance(kind, str) or kind not in INFORMATION_KINDS:
        raise ValueError(f"information_kind must be one of {INFORMATION_KINDS!r}")
    return {
        "information_kind": kind,
        **{name: _unit(features[name], name) for name in CONTINUOUS_FIELDS},
    }


def _finite_nonnegative(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (maximum is not None and value > maximum):
        bound = f" in [0, {maximum}]" if maximum is not None else " >= 0"
        raise ValueError(f"{name} must be finite and{bound}")
    return value


def validate_teaching_prior_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pre-query vector consumed by ``SyntheticGModel``.

    The model is a mechanism prior, so the declaration must be complete and
    explicit.  In particular, no action id, URL, title, or returned document
    is inspected to fill a missing feature.
    """
    if not isinstance(features, Mapping):
        raise ValueError("teaching prior features must be a mapping")
    keys = set(features)
    missing = set(TEACHING_REQUIRED_FIELDS) - keys
    extra = keys - set(TEACHING_REQUIRED_FIELDS)
    if missing:
        raise ValueError(f"teaching prior features missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"teaching prior features contain unknown fields: {sorted(extra)}")
    if features["schema"] != TEACHING_SCHEMA:
        raise ValueError(f"teaching prior features require schema {TEACHING_SCHEMA!r}")
    action = features["action"]
    scenario = features["scenario"]
    if action not in TEACHING_ACTIONS:
        raise ValueError(f"action must be one of {TEACHING_ACTIONS!r}")
    if scenario not in TEACHING_SCENARIOS:
        raise ValueError(f"scenario must be one of {TEACHING_SCENARIOS!r}")
    bounded = {
        name: _unit(features[name], name)
        for name in (
            "entropy", "max_probability", "source_reliability",
            "event_relevance", "tactical_sensitivity",
        )
    }
    bounded["time_to_kickoff_hours"] = _finite_nonnegative(
        features["time_to_kickoff_hours"], "time_to_kickoff_hours", maximum=168.0)
    bounded["cost_units"] = _finite_nonnegative(features["cost_units"], "cost_units")
    return {"schema": TEACHING_SCHEMA, "action": action, "scenario": scenario, **bounded}


def validate_any_prior_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Validate either versioned prior vector accepted by the runtime."""
    if isinstance(features, Mapping) and features.get("schema") == TEACHING_SCHEMA:
        return validate_teaching_prior_features(features)
    return validate_prior_features(features)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _probabilities(public_state: Mapping[str, Any]) -> tuple[float, ...]:
    """Read the frozen public prediction without looking at evidence text.

    The public runner has emitted both ``prediction.probabilities`` and (in
    older snapshots) a top-level ``probabilities`` projection.  Accept those
    two explicit locations only.  No fallback to a model file or a post-query
    document is allowed here.
    """
    prediction = public_state.get("prediction")
    values = prediction.get("probabilities") if isinstance(prediction, Mapping) else None
    if values is None:
        values = public_state.get("probabilities")
    if values is None:
        values = public_state.get("initial_probabilities")
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("public pre-query state requires at least two prediction probabilities")
    numbers = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probabilities[{index}] must be finite and nonnegative")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"probabilities[{index}] must be finite and nonnegative")
        numbers.append(number)
    total = math.fsum(numbers)
    if total <= 0.0:
        raise ValueError("public pre-query probabilities must have a positive sum")
    # A predictor may serialize a value rounded at the sixth decimal.  A
    # deterministic normalization handles that harmless representation detail
    # while rejecting vectors that are plainly not probabilities.
    normalized = tuple(value / total for value in numbers)
    return normalized


def _strict_public_state(public_state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the canonical public pre-query state used by builder v2.

    The historical v1 builder accepts a few legacy projections for replay.
    A new acquisition run must provide one unambiguous public snapshot.  The
    validator therefore rejects fallback aliases and enforces probabilities
    that are already normalized; it does not inspect any evidence/document
    fields that may be carried alongside the snapshot.
    """
    public_state = _mapping(public_state, "public_state")
    prediction = public_state.get("prediction")
    if not isinstance(prediction, Mapping):
        raise ValueError("strict public pre-query state requires prediction mapping")
    values = prediction.get("probabilities")
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("strict public pre-query state requires prediction.probabilities")
    numbers: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"prediction.probabilities[{index}] must be finite in [0,1]")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"prediction.probabilities[{index}] must be finite in [0,1]")
        numbers.append(number)
    if not math.isclose(math.fsum(numbers), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("strict prediction.probabilities must sum to 1")

    state = public_state.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("strict public pre-query state requires state mapping")
    base = state.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("strict public pre-query state requires state.base mapping")
    if "kickoff" not in base or "prediction_cutoff" not in state:
        raise ValueError("strict public pre-query state requires state.base.kickoff and state.prediction_cutoff")
    kickoff = _timestamp(base["kickoff"], "state.base.kickoff")
    cutoff = _timestamp(state["prediction_cutoff"], "state.prediction_cutoff")
    if cutoff >= kickoff:
        raise ValueError("strict prediction_cutoff must be before kickoff")

    # These aliases are accepted by v1 historical replay only.  Rejecting them
    # here prevents a caller from silently choosing a stale projection over the
    # canonical public snapshot.
    aliases = {
        "probabilities": public_state.get("probabilities"),
        "initial_probabilities": public_state.get("initial_probabilities"),
        "kickoff_at": public_state.get("kickoff_at"),
        "kickoff": public_state.get("kickoff"),
        "prediction_cutoff": public_state.get("prediction_cutoff"),
        "decision_at": public_state.get("decision_at"),
    }
    if any(value is not None for value in aliases.values()):
        raise ValueError("strict public pre-query state contains legacy fallback aliases")
    return public_state


def _timestamp(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _public_time(public_state: Mapping[str, Any], name: str) -> datetime:
    """Read a timestamp from declared public-state locations only."""
    candidates = []
    if name == "kickoff_at":
        candidates.extend((public_state.get("kickoff_at"), public_state.get("kickoff")))
        state = public_state.get("state")
        if isinstance(state, Mapping):
            base = state.get("base")
            if isinstance(base, Mapping):
                candidates.extend((base.get("kickoff_at"), base.get("kickoff")))
            candidates.extend((state.get("kickoff_at"), state.get("kickoff")))
    elif name == "prediction_cutoff":
        candidates.extend((public_state.get("prediction_cutoff"), public_state.get("decision_at")))
        state = public_state.get("state")
        if isinstance(state, Mapping):
            candidates.extend((state.get("prediction_cutoff"), state.get("decision_at")))
    for value in candidates:
        if value is not None:
            return _timestamp(value, name)
    raise ValueError(f"public pre-query state requires {name}")


def _capability_fields(capability: Mapping[str, Any], required: Sequence[str], name: str) -> dict[str, Any]:
    capability = _mapping(capability, name)
    missing = set(required) - set(capability)
    extra = set(capability) - set(required)
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} contains unknown fields: {sorted(extra)}")
    return dict(capability)


def _entropy_and_margin(probabilities: Sequence[float]) -> tuple[float, float, float]:
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)
    uncertainty = entropy / math.log(len(probabilities))
    ordered = sorted(probabilities, reverse=True)
    # A narrow top-two margin means the current decision is sensitive to new
    # information.  This is state geometry, not a claim about any source's
    # semantic relevance.
    margin_sensitivity = max(0.0, min(1.0, 1.0 - (ordered[0] - ordered[1])))
    return uncertainty, margin_sensitivity, ordered[0]


def build_prequery_prior_features(
    public_state: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    schema: str = SCHEMA,
) -> dict[str, Any]:
    """Build one prior vector from public state and explicit source capability.

    ``capability`` is a registry declaration supplied before the query.  It
    must name the information family and its source characteristics; the
    builder never derives those values from target text or URLs.  State
    quantities are deterministic functions of the current public prediction.
    The result remains a synthetic prior declaration and does not change the
    empirical UNKNOWN gate.
    """
    public_state = _mapping(public_state, "public_state")
    probabilities = _probabilities(public_state)
    uncertainty, margin_sensitivity, max_probability = _entropy_and_margin(probabilities)
    if schema == SCHEMA:
        declared = _capability_fields(capability, LEGACY_CAPABILITY_FIELDS, "capability")
        multiplier = _unit(declared["decision_sensitivity_multiplier"], "decision_sensitivity_multiplier")
        result = {
            "information_kind": declared["information_kind"],
            "uncertainty": uncertainty,
            "decision_sensitivity": max(0.0, min(1.0, margin_sensitivity * multiplier)),
            "source_reliability": declared["source_reliability"],
            "specificity": declared["specificity"],
            "redundancy": declared["redundancy"],
            "misdirection_risk": declared["misdirection_risk"],
            "query_cost": declared["query_cost"],
        }
        return validate_prior_features(result)
    if schema == TEACHING_SCHEMA:
        declared = _capability_fields(capability, TEACHING_CAPABILITY_FIELDS, "teaching capability")
        kickoff = _public_time(public_state, "kickoff_at")
        cutoff = _public_time(public_state, "prediction_cutoff")
        hours = (kickoff - cutoff).total_seconds() / 3600.0
        if hours < 0.0:
            raise ValueError("public kickoff must not precede prediction_cutoff")
        # The teaching artifact is bounded to one week ahead.  Reject rather
        # than silently clipping a schedule that violates its frozen contract.
        if hours > 168.0:
            raise ValueError("time_to_kickoff_hours must be at most 168")
        result = {
            "schema": TEACHING_SCHEMA,
            "action": declared["action"],
            "scenario": declared["scenario"],
            "entropy": uncertainty,
            "max_probability": max_probability,
            "time_to_kickoff_hours": hours,
            "source_reliability": declared["source_reliability"],
            "event_relevance": declared["event_relevance"],
            "tactical_sensitivity": declared["tactical_sensitivity"],
            "cost_units": declared["cost_units"],
        }
        return validate_teaching_prior_features(result)
    raise ValueError(f"unsupported pre-query feature schema: {schema!r}")


def build_strict_prequery_prior_features(
    public_state: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    schema: str = TEACHING_SCHEMA,
) -> dict[str, Any]:
    """Build a prior using the canonical, strictly pre-kickoff public state.

    This is the v2 online entrypoint.  It retains the same two feature
    schemas and explicit capability fields as v1, but rejects legacy state
    projections and non-normalized probabilities before deriving any feature.
    """
    checked = _strict_public_state(public_state)
    return build_prequery_prior_features(checked, capability, schema=schema)


def build_prequery_feature_map(
    public_state: Mapping[str, Any],
    capabilities_by_action: Mapping[str, Mapping[str, Any]],
    *,
    schema: str = SCHEMA,
) -> dict[str, dict[str, Any]]:
    """Build an exact action-id keyed map from a pre-query capability registry."""
    if not isinstance(capabilities_by_action, Mapping) or not capabilities_by_action:
        raise ValueError("capabilities_by_action must be a nonempty action-id mapping")
    result = {}
    for action_id, capability in capabilities_by_action.items():
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("capabilities_by_action keys must be nonempty action IDs")
        if action_id in result:
            raise ValueError(f"duplicate action_id: {action_id}")
        result[action_id] = build_prequery_prior_features(public_state, capability, schema=schema)
    return result


def build_strict_prequery_feature_map(
    public_state: Mapping[str, Any],
    capabilities_by_action: Mapping[str, Mapping[str, Any]],
    *,
    schema: str = TEACHING_SCHEMA,
) -> dict[str, dict[str, Any]]:
    """Build an exact action-id map under the strict v2 public-state contract."""
    _strict_public_state(public_state)
    if not isinstance(capabilities_by_action, Mapping) or not capabilities_by_action:
        raise ValueError("capabilities_by_action must be a nonempty action-id mapping")
    result: dict[str, dict[str, Any]] = {}
    for action_id, capability in capabilities_by_action.items():
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("capabilities_by_action keys must be nonempty action IDs")
        if action_id in result:
            raise ValueError(f"duplicate action_id: {action_id}")
        result[action_id] = build_strict_prequery_prior_features(
            public_state, capability, schema=schema)
    return result


def attach_prequery_prior_features(
    actions: Sequence[Mapping[str, Any]],
    public_state: Mapping[str, Any],
    capabilities_by_action: Mapping[str, Mapping[str, Any]],
    *,
    schema: str = SCHEMA,
    field: str = FIELD,
) -> list[dict[str, Any]]:
    """Build and attach prior vectors without reading action text or URLs."""
    declarations = build_prequery_feature_map(public_state, capabilities_by_action, schema=schema)
    return attach_any_prior_features(actions, declarations, field=field)


def attach_strict_prequery_prior_features(
    actions: Sequence[Mapping[str, Any]],
    public_state: Mapping[str, Any],
    capabilities_by_action: Mapping[str, Mapping[str, Any]],
    *,
    schema: str = TEACHING_SCHEMA,
    field: str = FIELD,
) -> list[dict[str, Any]]:
    """Attach v2 vectors after strict public-state and registry validation."""
    declarations = build_strict_prequery_feature_map(
        public_state, capabilities_by_action, schema=schema)
    return attach_any_prior_features(actions, declarations, field=field)


def attach_prior_features(
    actions: Sequence[Mapping[str, Any]],
    features_by_action: Mapping[str, Mapping[str, Any]],
    *,
    field: str = FIELD,
) -> list[dict[str, Any]]:
    """Attach explicit prior features to every action in a menu.

    ``features_by_action`` must cover the menu exactly.  Exact coverage makes
    missing declarations visible before the query and prevents a caller from
    accidentally relying on the adapter's unavailable fallback.  The helper
    never mutates actions supplied by a generator and never reads action text,
    URLs, or post-query documents.
    """
    if field not in {FIELD, ALIAS}:
        raise ValueError(f"field must be {FIELD!r} or {ALIAS!r}")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ValueError("actions must be a sequence of mappings")
    if not isinstance(features_by_action, Mapping):
        raise ValueError("features_by_action must be a mapping keyed by action_id")
    if any(not isinstance(key, str) or not key for key in features_by_action):
        raise ValueError("features_by_action keys must be non-empty action_id strings")

    ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("each action must be a mapping")
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("each action requires a non-empty action_id")
        if action_id in ids:
            raise ValueError(f"duplicate action_id: {action_id}")
        ids.append(action_id)
        if field in action or (field == FIELD and ALIAS in action) or (field == ALIAS and FIELD in action):
            raise ValueError(f"action {action_id} already carries prior features")

    declared = set(features_by_action)
    expected = set(ids)
    if declared != expected:
        missing = sorted(expected - declared)
        extra = sorted(declared - expected)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise ValueError("prior feature declarations do not exactly cover the menu (" + ", ".join(detail) + ")")

    result: list[dict[str, Any]] = []
    for action in actions:
        copy = deepcopy(dict(action))
        copy[field] = validate_prior_features(features_by_action[action["action_id"]])
        result.append(copy)
    return result


def attach_teaching_prior_features(
    actions: Sequence[Mapping[str, Any]],
    features_by_action: Mapping[str, Mapping[str, Any]],
    *,
    field: str = FIELD,
) -> list[dict[str, Any]]:
    """Attach exact, explicit ``synthetic-g-teaching-v1`` declarations."""
    if field not in {FIELD, ALIAS}:
        raise ValueError(f"field must be {FIELD!r} or {ALIAS!r}")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ValueError("actions must be a sequence of mappings")
    if not isinstance(features_by_action, Mapping):
        raise ValueError("features_by_action must be a mapping keyed by action_id")
    ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("each action must be a mapping")
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("each action requires a non-empty action_id")
        if action_id in ids:
            raise ValueError(f"duplicate action_id: {action_id}")
        ids.append(action_id)
        if field in action or (field == FIELD and ALIAS in action) or (field == ALIAS and FIELD in action):
            raise ValueError(f"action {action_id} already carries prior features")
    if set(features_by_action) != set(ids):
        missing = sorted(set(ids) - set(features_by_action))
        extra = sorted(set(features_by_action) - set(ids))
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise ValueError("teaching prior feature declarations do not exactly cover the menu (" + ", ".join(detail) + ")")
    result: list[dict[str, Any]] = []
    for action in actions:
        copy = deepcopy(dict(action))
        copy[field] = validate_teaching_prior_features(features_by_action[action["action_id"]])
        result.append(copy)
    return result


def attach_any_prior_features(
    actions: Sequence[Mapping[str, Any]],
    features_by_action: Mapping[str, Mapping[str, Any]],
    *,
    field: str = FIELD,
) -> list[dict[str, Any]]:
    """Dispatch to one complete feature contract, rejecting mixed versions."""
    values = list(features_by_action.values()) if isinstance(features_by_action, Mapping) else []
    teaching = [isinstance(value, Mapping) and value.get("schema") == TEACHING_SCHEMA for value in values]
    if any(teaching) and not all(teaching):
        raise ValueError("prior feature declarations must use one schema per menu")
    if any(teaching):
        return attach_teaching_prior_features(actions, features_by_action, field=field)
    return attach_prior_features(actions, features_by_action, field=field)


def strip_prior_features(action: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the two declared prior metadata fields for transport.

    Web executors bind the request identity (tool, URL/query, cost, and key),
    while these fields are decision-local annotations.  No other action field
    is stripped or ignored.
    """
    if not isinstance(action, Mapping):
        raise ValueError("action must be a mapping")
    return {key: deepcopy(value) for key, value in action.items()
            if key not in {FIELD, ALIAS}}


def prior_feature_manifest() -> dict[str, Any]:
    """Return the frozen public contract used by candidate producers."""
    return {
        "schema": SCHEMA,
        "prequery_builder_schema": PREQUERY_BUILDER_SCHEMA,
        "strict_prequery_builder_schema": STRICT_PREQUERY_BUILDER_SCHEMA,
        "field": FIELD,
        "alias": ALIAS,
        "required": list(REQUIRED_FIELDS),
        "information_kinds": list(INFORMATION_KINDS),
        "numeric_domain": "finite values in [0,1]",
        "prequery_state_inputs": [
            "prediction.probabilities (or explicit top-level probabilities)",
            "state.base.kickoff and state.prediction_cutoff for teaching time feature",
        ],
        "prequery_capability_inputs": {
            "bridge": list(LEGACY_CAPABILITY_FIELDS),
            "teaching": list(TEACHING_CAPABILITY_FIELDS),
        },
        "source_boundary": "pre_query_explicit_declaration_only",
        "post_reveal_inference": False,
        "selection_authority": "empirical_g_only",
        "accepted_versions": [SCHEMA, TEACHING_SCHEMA],
        "strict_public_state_contract": {
            "required": [
                "prediction.probabilities",
                "state.base.kickoff",
                "state.prediction_cutoff",
            ],
            "probabilities_must_sum_to": 1.0,
            "cutoff_must_be_before_kickoff": True,
            "legacy_fallback_aliases_rejected": True,
        },
    }


__all__ = [
    "SCHEMA",
    "PREQUERY_BUILDER_SCHEMA",
    "STRICT_PREQUERY_BUILDER_SCHEMA",
    "FIELD",
    "ALIAS",
    "INFORMATION_KINDS",
    "CONTINUOUS_FIELDS",
    "REQUIRED_FIELDS",
    "TEACHING_SCHEMA",
    "TEACHING_ACTIONS",
    "TEACHING_SCENARIOS",
    "TEACHING_REQUIRED_FIELDS",
    "LEGACY_CAPABILITY_FIELDS",
    "TEACHING_CAPABILITY_FIELDS",
    "validate_prior_features",
    "validate_teaching_prior_features",
    "validate_any_prior_features",
    "build_prequery_prior_features",
    "build_prequery_feature_map",
    "attach_prequery_prior_features",
    "build_strict_prequery_prior_features",
    "build_strict_prequery_feature_map",
    "attach_strict_prequery_prior_features",
    "attach_prior_features",
    "attach_teaching_prior_features",
    "attach_any_prior_features",
    "strip_prior_features",
    "prior_feature_manifest",
]
