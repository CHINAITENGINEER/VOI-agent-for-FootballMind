"""Opt-in, pre-selection observable-v3 diagnostics for generic runtime menus.

The wrapper uses the runtime's existing candidate_contexts hook. Actions,
their order, the empirical policy, and the transport contract are unchanged.
Only exact caller declarations are usable; absent values remain unavailable.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .g_prior_features import STRICT_PREQUERY_BUILDER_SCHEMA, _strict_public_state
from .learned_voi_runtime import digest
from football_research.acquisition.state import parse_time
from football_research.acquisition.synthetic_g_observable_v3 import (
    ObservableOnlyGModel, make_observable_features,
)

SCHEMA = "footballmind-observable-prequery-shadow-v1"
REGISTRY_SCHEMA = "footballmind-observable-prequery-registry-v1"
CONFIG_KEY = "observable_prequery_shadow"
MODEL_FILE = "observable-shadow-model.json"
DECLARATIONS = ("action", "source_reliability", "freshness", "redundancy",
                "response_probability", "updater_trust", "cost_units")
PROVENANCE_FIELDS = ("declared_at", "basis", "request_binding_sha256")


def request_binding(action):
    # Identity binding only; none of these request bytes become model features.
    from .g_prior_features import strip_prior_features
    return digest(strip_prior_features(action))


def canonical_projection(public):
    """Opt in to v2 using canonical slots, without interpreting legacy aliases.

    Runtime decision_at is an audit clock, not a prediction-cutoff fallback.
    The original envelope is retained in public_before by the runtime.
    """
    state = public.get("state") or {}
    base = state.get("base") or {}
    prediction = public.get("prediction")
    return {
        "prediction": ({"probabilities": deepcopy(prediction.get("probabilities"))}
                       if isinstance(prediction, dict) else None),
        "state": {"base": {"kickoff": base.get("kickoff")},
                  "prediction_cutoff": state.get("prediction_cutoff")},
    }


def build_observable_menu(public_state, actions, *, registry, model, decision_at):
    """Return an exact ordered action map; never infer a missing declaration."""
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Explicit observable pre-query registry schema required")
    declarations = registry.get("capabilities_by_action_id")
    if not isinstance(declarations, dict):
        raise ValueError("capabilities_by_action_id must be a mapping")
    ids = [action["action_id"] for action in actions]
    if any(not isinstance(aid, str) or not aid for aid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Menu action_ids must be nonempty and unique")
    state_error, hours = None, None
    try:
        _strict_public_state(public_state)
        state = public_state["state"]
        kickoff, cutoff = map(parse_time, (state["base"]["kickoff"], state["prediction_cutoff"]))
        decision = parse_time(decision_at)
        if decision >= kickoff or decision > cutoff:
            raise ValueError("Actual decision must precede kickoff and not exceed cutoff")
        hours = (kickoff - cutoff).total_seconds() / 3600.
    except (ValueError, TypeError, KeyError) as exc:
        state_error = str(exc)
    rows, rankable = [], []
    for action in actions:
        aid = action["action_id"]
        declaration = declarations.get(aid)
        missing, errors = [], []
        if declaration is None:
            missing.append("explicit_action_id_registry_entry")
            declaration = {}
        if not isinstance(declaration, dict):
            errors.append("declaration_must_be_a_mapping")
            declaration = {}
        missing += [name for name in (*DECLARATIONS, *PROVENANCE_FIELDS)
                    if declaration.get(name) is None]
        extra = set(declaration) - set(DECLARATIONS) - set(PROVENANCE_FIELDS)
        if extra:
            errors.append("undeclared_fields:" + ",".join(sorted(extra)))
        if state_error:
            errors.append("strict_public_state:" + state_error)
        if not missing and not errors:
            try:
                if not isinstance(declaration["basis"], str) or not declaration["basis"].strip():
                    raise ValueError("Explicit declaration basis required")
                if parse_time(declaration["declared_at"]) > parse_time(decision_at):
                    raise ValueError("Declaration was not available before selection")
                if declaration["request_binding_sha256"] != request_binding(action):
                    raise ValueError("Declaration does not bind this exact request/cost")
                features = make_observable_features({
                    "probabilities": list(public_state["prediction"]["probabilities"]),
                    "time_to_kickoff_hours": hours,
                    **{key: declaration[key] for key in DECLARATIONS if key != "cost_units"},
                })
                estimate = model.estimate(features, cost_units=declaration["cost_units"])
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
        available = not missing and not errors
        row = {
            "action_id": aid, "request_binding_sha256": request_binding(action),
            "status": "AVAILABLE_DIAGNOSTIC_ONLY" if available else "UNAVAILABLE",
            "diagnostic_only": True, "selection_allowed": False,
            "declared_capability": deepcopy(declaration),
            "missing_fields": missing, "validation_errors": errors,
            "features": features if available else None,
            "synthetic_estimate": estimate if available else None,
            "empirical_expected_signed_gain": None, "empirical_net_voi": None,
        }
        rows.append(row)
        if available:
            rankable.append({"action_id": aid, "features": features,
                             "cost_units": declaration["cost_units"]})
    ranking = model.rank_shadow(rankable) if rankable else None
    complete = bool(actions) and len(rankable) == len(actions)
    return {
        "schema": SCHEMA, "strict_public_state_contract": STRICT_PREQUERY_BUILDER_SCHEMA,
        "decision_at": decision_at, "canonical_public_state": canonical_projection(public_state),
        "canonical_time_to_kickoff_hours": hours,
        "time_basis": "state.base.kickoff_minus_state.prediction_cutoff_not_wall_clock",
        "menu_action_ids": ids, "by_action_id": {r["action_id"]: r for r in rows},
        "registry_sha256": digest(registry),
        "registry_entries_outside_current_menu": sorted(set(declarations) - set(ids)),
        "rankable_candidates": len(rankable), "unavailable_candidates": len(rows) - len(rankable),
        "coverage_complete": complete, "subset_ranking": ranking,
        "shadow_proposal": ranking["shadow_proposal"] if complete else None,
        "selection_allowed": False, "empirical_gate_modified": False,
        "unavailable_is_not_stop": True,
    }


class ObservablePrequeryShadow:
    """Decorate a generator without adding new action/transport fields."""

    def __init__(self, generator, *, registry, model_bytes, expected_model_sha256):
        if sha256(model_bytes).hexdigest() != expected_model_sha256:
            raise ValueError("Observable model raw artifact digest mismatch")
        if registry.get("schema") != REGISTRY_SCHEMA or not isinstance(registry.get("capabilities_by_action_id"), dict):
            raise ValueError("Explicit observable registry required")
        self.generator = generator
        self.registry = deepcopy(registry)
        self.model_bytes = bytes(model_bytes)
        self.model = ObservableOnlyGModel(json.loads(self.model_bytes))
        self.model_raw_sha256 = expected_model_sha256
        self.version = generator.version

    def __call__(self, public):
        return self.generator(public)

    def configuration(self):
        return {**self.generator.configuration(), CONFIG_KEY: {
            "schema": SCHEMA, "registry": deepcopy(self.registry),
            "model_raw_sha256": self.model_raw_sha256,
            "model_canonical_sha256": self.model.model_sha256,
            "selection_allowed": False, "hook": "candidate_contexts_before_policy_plan",
            "public_state_projection": "canonical_slots_only_v1",
        }}

    def candidate_contexts(self, public, actions):
        projector = getattr(self.generator, "candidate_contexts", None)
        context = (projector(deepcopy(public), deepcopy(actions)) if callable(projector) else None) or {}
        result = build_observable_menu(canonical_projection(public), actions,
            registry=self.registry, model=self.model, decision_at=public.get("decision_at"))
        result["model_raw_sha256"] = self.model_raw_sha256
        result["model_canonical_sha256"] = self.model.model_sha256
        return {**context, CONFIG_KEY: result}

    @classmethod
    def from_configuration(cls, generator, config, model_path):
        wrapped = cls(generator, registry=config["registry"], model_bytes=Path(model_path).read_bytes(),
                      expected_model_sha256=config["model_raw_sha256"])
        if wrapped.configuration()[CONFIG_KEY] != config:
            raise ValueError("Observable pre-query shadow configuration changed")
        return wrapped


def load_shadow_spec(path):
    """Load and pin declarations/model before any provider I/O."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise ValueError("Versioned observable shadow spec required")
    model_bytes = Path(spec["model_path"]).read_bytes()
    if sha256(model_bytes).hexdigest() != spec["model_raw_sha256"]:
        raise ValueError("Observable shadow model file changed")
    # Validate model and registry even if a later fixture check blocks execution.
    ObservableOnlyGModel(json.loads(model_bytes))
    if spec["registry"].get("schema") != REGISTRY_SCHEMA or not isinstance(spec["registry"].get("capabilities_by_action_id"), dict):
        raise ValueError("Explicit observable registry required")
    return spec, model_bytes
