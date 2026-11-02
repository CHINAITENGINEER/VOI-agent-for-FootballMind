"""Strict, explicitly supplied history for the runtime VoI selection boundary.

The package records provenance declarations, not independent verification.  It
ships no experience and grants no permission to use synthetic support in a live
run.  The selection policy must enforce that separate deployment boundary.
``decision_loss`` denotes a declared normalized loss in [0, 1], so its signed
before/after difference lies in [-1, 1]. Unbounded log loss and unnormalized
Brier loss are outside this package's Hoeffding contract. Independence of the
declared episode/dependency clusters remains a calibration-manifest assumption.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from string import Formatter
from urllib.parse import urlsplit

from .state_bound_context_value import StateBoundContextValueEstimator
from .voi_evidence_bridge import VoIEvidenceLink, VERSION as LINK_VERSION


VERSION = "runtime-voi-model-v1"
SOURCE_CLAIM_VERSION = "runtime-voi-model-v2"
SOURCE_CLAIM_SEMANTICS = "actor_source_claims_not_world_facts_v1"
SOURCE_SUPPORT_VERSION = "runtime-voi-model-v3"
CONTEXT_KEYS = frozenset({
    "operation_document_count", "operation_readable_document_count",
    "operation_last_return_surface", "qualification_state", "tactical_layout",
})
SOURCE_CLAIM_CONTEXT_KEYS = frozenset({
    "home_reported_qualification_state", "away_reported_qualification_state",
    "home_reported_tactical_layout", "away_reported_tactical_layout",
})
SOURCE_CLAIM_MODEL_CONTEXT_KEYS = (
    CONTEXT_KEYS - {"qualification_state", "tactical_layout"}
) | SOURCE_CLAIM_CONTEXT_KEYS
LOSS_UNITS = frozenset({"decision_loss"})
_CONFIG_KEYS = frozenset({"version", "evidence_basis", "calibration_as_of", "loss_unit",
                          "context_keys", "min_support", "confidence", "family_size", "routes", "history"})
_ROUTE_KEYS = frozenset({"route_id", "match", "source_id", "source_version", "action_id", "cost", "unit"})
_LINK_KEYS = frozenset(field.name for field in fields(VoIEvidenceLink)) | {"schema_version"}
_LINK_STATUSES = frozenset({"linked_available", "linked_unusable", "missing_evidence", "ambiguous_evidence"})
_TARGET = re.compile(r"spftll:[1-9][0-9]*\Z")


def _object(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name}: exact fields required")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name}: nonempty unpadded string required")
    return value


def _time(value, name):
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name}: timestamp required") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name}: timezone required")
    return parsed.astimezone(timezone.utc)


def _number(value, name, minimum=None, maximum=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name}: finite number required")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name}: outside allowed range")
    return value


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")
    return value


def _json(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("configuration keys must be strings")
        for child in value.values():
            _json(child)
    elif type(value) is list:
        for child in value:
            _json(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("configuration must contain only JSON values")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("configuration must contain only finite numbers")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _route_match(value):
    if type(value) is not dict:
        raise ValueError("route.match: object required")
    action = value.get("action")
    if action == "read_url":
        _object(value, {"action", "url"}, "route.match")
        url = _text(value["url"], "route.match.url")
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and not parsed.username and not parsed.password and not parsed.fragment)
            parsed.port
        except ValueError as exc:
            raise ValueError("route.match.url: invalid URL") from exc
        if not valid or any(char.isspace() for char in url):
            raise ValueError("route.match.url: explicit HTTP URL required")
    elif action == "search_web":
        _object(value, {"action", "engine", "query_template"}, "route.match")
        if type(value["engine"]) is not str or value["engine"] not in {"bing", "duckduckgo", "news"}:
            raise ValueError("route.match.engine: unsupported engine")
        template = _text(value["query_template"], "route.match.query_template")
        literal = template.replace("{match_input}", "")
        if "{" in literal or "}" in literal:
            raise ValueError("route.match.query_template: only literal text and {match_input} are permitted")
        try:
            parts = list(Formatter().parse(template))
        except ValueError as exc:
            raise ValueError("route.match.query_template: invalid template") from exc
        if any(field not in (None, "match_input") or spec or conversion
               for _, field, spec, conversion in parts):
            raise ValueError("route.match.query_template: only plain {match_input} is permitted")
    else:
        raise ValueError("route.match.action: unsupported action")


class RuntimeVoIModel:
    version = VERSION

    def __init__(self, config):
        _json(config)
        source_support = type(config) is dict and config.get("version") == SOURCE_SUPPORT_VERSION
        source_claims = type(config) is dict and config.get("version") in (SOURCE_CLAIM_VERSION, SOURCE_SUPPORT_VERSION)
        required = _CONFIG_KEYS | ({"context_semantics"} if source_claims else set())
        _object(config, required | ({"source_context_contract"} if source_support else set()), "configuration")
        if config["version"] not in (VERSION, SOURCE_CLAIM_VERSION, SOURCE_SUPPORT_VERSION):
            raise ValueError("runtime VoI model version mismatch")
        if source_support:
            from .runtime_source_support import configuration, SEMANTICS
            if config["context_semantics"] != SEMANTICS:
                raise ValueError("context_semantics: structured fixture source semantics required")
            configuration(config["source_context_contract"])
        elif source_claims and config["context_semantics"] != SOURCE_CLAIM_SEMANTICS:
            raise ValueError("context_semantics: actor source claim semantics required")
        if type(config["evidence_basis"]) is not str or config["evidence_basis"] not in {"synthetic", "independently_evaluated"}:
            raise ValueError("unsupported evidence_basis")
        if type(config["loss_unit"]) is not str or config["loss_unit"] not in LOSS_UNITS:
            raise ValueError("unsupported loss_unit")
        calibrated = _time(config["calibration_as_of"], "calibration_as_of")
        keys = config["context_keys"]
        permitted_keys = SOURCE_CLAIM_MODEL_CONTEXT_KEYS if source_claims else CONTEXT_KEYS
        if (type(keys) is not list or not keys or any(type(key) is not str for key in keys)
                or len(keys) != len(set(keys)) or set(keys) - permitted_keys):
            raise ValueError("context_keys: unique supported fields required")
        _integer(config["min_support"], "min_support", 1)
        _integer(config["family_size"], "family_size", 1)
        confidence = _number(config["confidence"], "confidence")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if type(config["routes"]) is not list or type(config["history"]) is not list:
            raise ValueError("routes and history must be lists")
        routes, matchers = {}, set()
        for route in config["routes"]:
            _object(route, _ROUTE_KEYS, "route")
            for key in ("route_id", "source_id", "source_version", "action_id", "unit"):
                _text(route[key], "route." + key)
            if route["route_id"] in routes:
                raise ValueError("duplicate route_id")
            _route_match(route["match"])
            matcher = json.dumps(route["match"], sort_keys=True)
            if matcher in matchers:
                raise ValueError("duplicate exact route matcher")
            matchers.add(matcher)
            _number(route["cost"], "route.cost", 0)
            if route["unit"] != config["loss_unit"]:
                raise ValueError("route unit differs from loss_unit")
            routes[route["route_id"]] = route
        if config["family_size"] < len(routes):
            raise ValueError("family_size must cover registered routes")
        links, contexts, link_ids = [], {}, set()
        for row in config["history"]:
            _object(row, {"link", "context", "label_available_at"}, "history row")
            raw = _object(row["link"], _LINK_KEYS, "history link")
            if raw["schema_version"] != LINK_VERSION:
                raise ValueError("history link version mismatch")
            for key in ("link_id", "trace_id", "episode_id", "target_key", "request_id",
                        "action_id", "candidate_id", "source_id", "source_version", "status"):
                _text(raw[key], "history link." + key)
            for key in ("evidence_entry_id", "evidence_availability", "evidence_quality", "dependency_key"):
                if raw[key] is not None:
                    _text(raw[key], "history link." + key)
            if raw["link_id"] in link_ids:
                raise ValueError("duplicate link_id")
            link_ids.add(raw["link_id"])
            if not _TARGET.fullmatch(raw["target_key"]):
                raise ValueError("history target_key must be canonical spftll:ID")
            route = routes.get(raw["candidate_id"])
            if route is None or any(raw[key] != route[key] for key in ("source_id", "source_version", "action_id")):
                raise ValueError("history link route mismatch")
            cutoff = _time(raw["prediction_cutoff"], "history prediction_cutoff")
            requested = _time(raw["requested_at"], "history requested_at")
            available = _time(row["label_available_at"], "label_available_at")
            if not cutoff <= available <= calibrated or requested > available:
                raise ValueError("history label/request chronology mismatch")
            if raw["status"] not in _LINK_STATUSES:
                raise ValueError("unsupported history link status")
            gain = raw["observed_risk_reduction"]
            if gain is not None:
                _number(gain, "observed_risk_reduction", -1, 1)
            for key in ("expected_information_value", "calibration_error"):
                if raw[key] is not None:
                    _number(raw[key], key)
            if raw["status"] == "linked_available":
                if (gain is None or raw["evidence_entry_id"] is None
                        or raw["evidence_availability"] != "available"
                        or raw["evidence_quality"] not in {"VALIDATED", "PARTIAL"}
                        or requested > cutoff):
                    raise ValueError("available history needs measured usable evidence")
            elif gain not in (None, 0):
                raise ValueError("unusable history cannot carry nonzero risk reduction")
            context = _object(row["context"], keys, "history context")
            for key, value in context.items():
                if key in {"operation_document_count", "operation_readable_document_count"}:
                    _integer(value, "history context." + key, 0)
                elif value is not None:
                    _text(value, "history context." + key)
            if ("operation_document_count" in context and "operation_readable_document_count" in context
                    and context["operation_readable_document_count"] > context["operation_document_count"]):
                raise ValueError("readable document count exceeds document count")
            link = VoIEvidenceLink(**{key: value for key, value in raw.items() if key != "schema_version"})
            links.append(link)
            contexts[link.link_id] = deepcopy(context)
        self._config = deepcopy(config)
        self.version = config["version"]
        self._links, self._contexts = tuple(links), contexts
        self._calibration_as_of = calibrated.isoformat()
        packed = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self._hash = hashlib.sha256(packed.encode("utf-8")).hexdigest()

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object))

    @property
    def evidence_basis(self):
        return self._config["evidence_basis"]

    @property
    def loss_unit(self):
        return self._config["loss_unit"]

    @property
    def calibration_as_of(self):
        return self._calibration_as_of

    @property
    def context_keys(self):
        return tuple(self._config["context_keys"])

    @property
    def context_semantics(self):
        return self._config.get("context_semantics")

    @property
    def source_context_contract(self):
        return deepcopy(self._config.get("source_context_contract"))

    @property
    def routes(self):
        return tuple(deepcopy(self._config["routes"]))

    @property
    def estimator(self):
        routes = {row["route_id"]: {key: row[key] for key in ("source_id", "source_version", "action_id", "cost", "unit")}
                  for row in self._config["routes"]}
        return StateBoundContextValueEstimator(
            self._links, deepcopy(self._contexts), routes, context_keys=self.context_keys,
            min_support=self._config["min_support"], confidence=self._config["confidence"],
            calibration_as_of=self.calibration_as_of, family_size=self._config["family_size"])

    def configuration(self):
        descriptor = {"adapter_version": self.version, "model_hash": self._hash,
                "evidence_basis": self.evidence_basis, "evidence_basis_verified": False,
                "numeric_voi_calibrated": False, "synthetic_live_use_permitted": False,
                "loss_unit": self.loss_unit, "calibration_as_of": self.calibration_as_of,
                "context_keys": list(self.context_keys), "min_support": self._config["min_support"],
                "confidence": self._config["confidence"], "family_size": self._config["family_size"],
                "routes": list(self.routes), "history_count": len(self._links),
                "loss_semantics": "declared_normalized_loss_in_[0,1];signed_reduction_in_[-1,1]",
                "cluster_independence": "manifest_assumption_not_independently_verified",
                "support_semantics": "declared_history; independent_evaluation_not_verified_by_loader"}
        if self.version in (SOURCE_CLAIM_VERSION, SOURCE_SUPPORT_VERSION):
            descriptor["context_semantics"] = self.context_semantics
        if self.version == SOURCE_SUPPORT_VERSION:
            descriptor["source_context_contract"] = self.source_context_contract
        return descriptor


__all__ = ["VERSION", "CONTEXT_KEYS", "SOURCE_CLAIM_VERSION", "SOURCE_CLAIM_SEMANTICS", "SOURCE_SUPPORT_VERSION",
           "SOURCE_CLAIM_CONTEXT_KEYS", "SOURCE_CLAIM_MODEL_CONTEXT_KEYS", "RuntimeVoIModel"]
