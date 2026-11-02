"""Exact provider-identity continuity for contextual-value sensitivity analysis.

The crosswalk is offline reconstructed metadata, not evidence that a mapping
was verified at the historical prediction time. It changes only the value
model's public categorical projection, never outcome models or OOF labels.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .contextual_value import ContextualValueEstimator, VERSION as CORE_VERSION
from .state import sanitize_state
from .value import _canonical

VERSION = "contextual-source-identity-v1"


def build_crosswalk(bundles, provider="understat"):
    if provider != "understat":
        raise ValueError("This frozen sensitivity study accepts Understat only")
    aliases, sources = {"team": {}, "competition": {}}, []
    for bundle in bundles:
        path = Path(bundle)/"id_mappings.jsonl"
        active = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            kind = row.get("entity_type")
            if row.get("source") != provider or kind not in aliases:
                continue
            if row.get("status") not in {"confirmed", "source_scoped"}:
                raise ValueError("Crosswalk rejects disputed, rejected, or undeclared mapping status")
            if row.get("valid_to") is not None:
                raise ValueError("Crosswalk export contains expired mapping")
            sid, entity_id = str(row["source_id"]), str(row["entity_id"])
            if not sid or not entity_id or (kind == "team" and not sid.isdigit()) or (kind == "competition" and sid != "EPL"):
                raise ValueError("Unexpected source identity")
            source_key = (kind, sid)
            if source_key in active:
                raise ValueError("Ambiguous repeated provider identity within export")
            active.add(source_key)
            token = f"provider:{provider}:{kind}:{sid}"
            if entity_id in aliases[kind] and aliases[kind][entity_id] != token:
                raise ValueError("Canonical UUID maps to conflicting provider identities")
            aliases[kind][entity_id] = token
        sources.append({"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
                        "mapping_keys": len(active)})
    payload = {"provider": provider, "aliases": aliases, "sources": sources,
        "construction": "exact_provider_entity_type_source_id_only_no_names_outcomes_or_numeric_features",
        "time_interpretation": "offline_reconstructed_identity_not_historical_verification",
        "unmapped_policy": "preserve_unknown_identifier_and_report_not_fuzzy_match"}
    payload["version"] = "provider-identity-crosswalk:"+sha256(_canonical(payload).encode()).hexdigest()[:20]
    return payload


def validate_crosswalk(crosswalk):
    payload = {k: v for k, v in crosswalk.items() if k != "version"}
    if crosswalk.get("version") != "provider-identity-crosswalk:"+sha256(_canonical(payload).encode()).hexdigest()[:20]:
        raise ValueError("Crosswalk hash mismatch")
    if crosswalk.get("provider") != "understat" or set(crosswalk.get("aliases", {})) != {"team", "competition"}:
        raise ValueError("Unsupported crosswalk provider/schema")
    for kind, aliases in crosswalk["aliases"].items():
        for identifier, token in aliases.items():
            if not identifier or not token.startswith(f"provider:understat:{kind}:"):
                raise ValueError("Invalid crosswalk alias")


class SourceIdentityValueEstimator:
    def __init__(self, crosswalk, regularization=20.0, min_support=12, risk_multiplier=1.64):
        validate_crosswalk(crosswalk)
        self.crosswalk = json.loads(json.dumps(crosswalk))
        self.model = ContextualValueEstimator(mode="visible-context", regularization=regularization,
                                             min_support=min_support, risk_multiplier=risk_multiplier)
        self.version = VERSION

    @property
    def min_support(self):
        return self.model.min_support

    @property
    def manifest(self):
        return self.model.manifest

    @property
    def calibration(self):
        return self.model.calibration

    @property
    def mode(self):
        return "source-identity"

    def project(self, state):
        clean = sanitize_state(state)
        mapped, unmapped = [], []
        for field, kind in (("home_team_id", "team"), ("away_team_id", "team"), ("competition_id", "competition")):
            original = clean["base"][field]
            replacement = self.crosswalk["aliases"][kind].get(original)
            if replacement is None:
                unmapped.append(field)
            else:
                clean["base"][field] = replacement
                mapped.append(field)
        return clean, {"mapped_fields": mapped, "unmapped_fields": unmapped, "crosswalk_version": self.crosswalk["version"]}

    def fit(self, samples, *, as_of, dataset_version, calibration_samples=None, calibration_target_coverage=.8, metadata=None):
        def project_rows(rows):
            return [{**row, "state": self.project(row["state"])[0]} for row in rows]
        self.model.fit(project_rows(samples), as_of=as_of, dataset_version=dataset_version,
            calibration_samples=project_rows(calibration_samples or []), calibration_target_coverage=calibration_target_coverage,
            metadata={**(metadata or {}), "identity_crosswalk_version": self.crosswalk["version"],
                      "identity_time_interpretation": self.crosswalk["time_interpretation"]})
        self.version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def estimate_value(self, visible_state, candidates, prediction=None, risk_level="mean"):
        public, audit = self.project(visible_state)
        estimates = self.model.estimate_value(public, candidates, prediction=prediction, risk_level=risk_level)
        return [{**e, "version": self.version, "estimator_version": self.version, "mode": self.mode,
                 "identity_projection": audit} for e in estimates]

    def _payload(self):
        return {"schema": VERSION, "crosswalk": self.crosswalk,
                "core": {**self.model._payload(), "version": self.model.version}}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self._payload(), "version": self.version}, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported source-identity value artifact")
        core = payload["core"]
        parameters = {k: v for k, v in core["parameters"].items() if k != "mode"}
        model = cls(payload["crosswalk"], **parameters)
        if core.get("schema") != CORE_VERSION or core["parameters"].get("mode") != "visible-context":
            raise ValueError("Unsupported contextual core")
        model.model.cells, model.model.manifest, model.model.calibration = core["cells"], core["manifest"], core["calibration"]
        core_version = CORE_VERSION+":"+sha256(_canonical(model.model._payload()).encode()).hexdigest()[:20]
        if core_version != core.get("version"):
            raise ValueError("Core artifact hash mismatch")
        model.model.version = core_version
        version = VERSION+":"+sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if version != payload.get("version"):
            raise ValueError("Identity model artifact hash mismatch")
        model.version = version
        return model
