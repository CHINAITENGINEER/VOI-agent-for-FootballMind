"""Exact, auditable joins of frozen prematch feature snapshots.

This module consumes platform features; it does not calculate team ratings or
parse source responses. Missing snapshots are exclusions, never nearest-time
substitutions. JSONL and the existing FeatureStore Parquet layout are accepted.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping


DEFAULT_FEATURES = frozenset({"home_elo", "away_elo", "elo_difference"} | {
    f"{side}_{field}" for side in ("home", "away") for field in (
        "form_points", "form_goals_for", "form_goals_against", "form_matches",
        "rest_days", "rolling_xg_for", "rolling_xg_against", "rolling_xg_difference",
        "rolling_goals_for", "rolling_goals_against", "rolling_points", "rolling_matches")})
KEY_FIELDS = ("match_id", "prediction_cutoff", "information_mode", "feature_version")


def _utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Feature timestamps require explicit timezone")
    return value.astimezone(timezone.utc)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda x: _utc(x).isoformat(), allow_nan=False)


def _hash(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _safe_feature(name):
    lowered = name.lower()
    if any(word in lowered for word in ("forecast", "target", "outcome", "score", "final", "season_total", "result")):
        return False
    if lowered in {"home_goals", "away_goals", "home_xg", "away_xg", "xg", "rank", "season_points"}:
        return False
    if "xg" in lowered and not any(word in lowered for word in ("rolling", "history", "form", "lag")):
        return False
    return True


class FeatureJoinError(ValueError):
    """A machine-readable exclusion reason for a single requested snapshot."""
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail
        super().__init__(reason + (": " + str(detail) if detail is not None else ""))


class FeatureIndex:
    """Index by canonical match ID, exact cutoff, mode and feature version.

    Aliases require ``{source_id: {target_id: canonical_id, evidence: ...}}``.
    Caller-specified numeric feature names extend the conservative defaults;
    target labels and unqualified match xG remain prohibited.
    """
    def __init__(self, *, aliases=None, allowed_features=None):
        self.allowed_features = DEFAULT_FEATURES | frozenset(allowed_features or ())
        unsafe = sorted(name for name in self.allowed_features if not _safe_feature(name))
        if unsafe:
            raise ValueError(f"Disallowed target/final feature names: {unsafe}")
        self.aliases = deepcopy(aliases or {})
        for source, mapping in self.aliases.items():
            if not isinstance(mapping, Mapping) or not mapping.get("target_id") or not mapping.get("evidence"):
                raise ValueError(f"ID alias {source!r} requires target_id and evidence")
            target = str(mapping["target_id"])
            if target in self.aliases and target != source:
                raise ValueError("Alias chains are ambiguous; map directly to canonical IDs")
        self._rows, self._conflicts = {}, set()
        self.rejections, self.duplicates = [], 0
        self.metadata, self.sources = [], []

    def _id(self, value):
        value = str(value)
        return str(self.aliases[value]["target_id"]) if value in self.aliases else value

    @classmethod
    def from_rows(cls, rows, *, aliases=None, allowed_features=None, strict=False):
        result = cls(aliases=aliases, allowed_features=allowed_features)
        for number, row in enumerate(rows, 1):
            if row.get("kind") == "metadata":
                result.metadata.append(deepcopy(row))
                continue
            try:
                normalized = result._normalize(row)
                key = tuple(normalized[name] for name in KEY_FIELDS)
                if key in result._conflicts:
                    raise FeatureJoinError("conflicting_duplicate_snapshot", list(map(str, key)))
                previous = result._rows.get(key)
                if previous is not None:
                    if previous["snapshot_sha256"] != normalized["snapshot_sha256"]:
                        del result._rows[key]
                        result._conflicts.add(key)
                        raise FeatureJoinError("conflicting_duplicate_snapshot", list(map(str, key)))
                    result.duplicates += 1
                else:
                    result._rows[key] = normalized
            except (ValueError, KeyError, TypeError) as exc:
                if strict:
                    raise
                result.rejections.append({"row": number, "match_id": row.get("match_id"),
                                          "reason": getattr(exc, "reason", str(exc))})
        return result

    def _normalize(self, original):
        row = deepcopy(dict(original))
        manifest = row.get("manifest", row.get("input_manifest"))
        if "manifest_json" in row:
            manifest = json.loads(row["manifest_json"])
        if not isinstance(manifest, (Mapping, list, tuple)) or not manifest:
            raise FeatureJoinError("missing_feature_manifest")
        expected = row.get("input_manifest_hash", row.get("manifest_hash"))
        if not expected or expected != _hash(manifest):
            raise FeatureJoinError("feature_manifest_hash_mismatch")
        # Store returns identity within manifest; Parquet/flat exports repeat it.
        if isinstance(manifest, Mapping):
            for field in KEY_FIELDS:
                if field not in row and field in manifest:
                    row[field] = manifest[field]
                if field in manifest:
                    lhs, rhs = row[field], manifest[field]
                    if field == "prediction_cutoff":
                        lhs, rhs = _utc(lhs), _utc(rhs)
                    if lhs != rhs:
                        raise FeatureJoinError("manifest_identity_mismatch", field)
        for field in KEY_FIELDS:
            if not row.get(field):
                raise FeatureJoinError("missing_snapshot_key", field)
        cutoff = _utc(row["prediction_cutoff"])
        mode = row["information_mode"]
        if mode not in {"operational", "historical_reconstruction"}:
            raise FeatureJoinError("unsupported_information_mode")
        available = row.get("features_available_at")
        if available is None and isinstance(manifest, Mapping):
            available = manifest.get("features_available_at")
            if available is None and manifest.get("availability_basis") in {
                "max(observed_at,ingested_at)", "trusted_provider_snapshot_plus_assumed_delay"}:
                available = cutoff  # Verified platform logical input upper bound.
        if available is None:
            raise FeatureJoinError("unknown_feature_availability")
        available = _utc(available)
        if available > cutoff:
            raise FeatureJoinError("future_feature_availability")
        for name in ("elo_available_at", "xg_available_at", "max_input_available_at"):
            if row.get(name) is not None and _utc(row[name]) > cutoff:
                raise FeatureJoinError("future_feature_availability", name)
        if row.get("kickoff") is not None and cutoff >= _utc(row["kickoff"]):
            raise FeatureJoinError("feature_cutoff_not_prematch")
        canonical_id = self._id(row["match_id"])
        if isinstance(manifest, Mapping):
            for name in ("max_input_available_at", "features_available_at", "elo_available_at", "xg_available_at"):
                if manifest.get(name) is not None and _utc(manifest[name]) > cutoff:
                    raise FeatureJoinError("future_feature_history", "manifest." + name)
            history_ids = manifest.get("history_match_ids", [])
            if any(self._id(mid) == canonical_id for mid in history_ids):
                raise FeatureJoinError("target_match_in_feature_history")
            history = [*manifest.get("history_matches", []), *manifest.get("history_inputs", []),
                       *manifest.get("xg_inputs", [])]
            for item in history:
                if not isinstance(item, Mapping):
                    continue
                if item.get("match_id") is not None and self._id(item["match_id"]) == canonical_id:
                    raise FeatureJoinError("target_match_in_feature_history")
                for name in ("available_at", "result_available_at", "features_available_at", "xg_available_at", "elo_available_at"):
                    if item.get(name) is not None and _utc(item[name]) > cutoff:
                        raise FeatureJoinError("future_feature_history", name)
                if item.get("kickoff") is not None and _utc(item["kickoff"]) >= cutoff:
                    raise FeatureJoinError("future_feature_history", "kickoff")
                if mode == "operational":
                    for name in ("observed_at", "ingested_at"):
                        if item.get(name) is not None and _utc(item[name]) > cutoff:
                            raise FeatureJoinError("future_operational_observation", name)
        values = row.get("features", row)
        if "features_json" in row:
            values = json.loads(row["features_json"])
        values = deepcopy(dict(values))
        elo = values.get("elo")
        if isinstance(elo, Mapping):
            values.update(home_elo=elo.get("home"), away_elo=elo.get("away"))
            if elo.get("difference") is not None:
                values["elo_difference"] = elo["difference"]
            if isinstance(manifest, Mapping) and "history_version_ids" in manifest:
                if sorted(elo.get("input_version_ids", [])) != sorted(manifest["history_version_ids"]):
                    raise FeatureJoinError("elo_lineage_mismatch")
                if elo.get("parameters") != manifest.get("parameters"):
                    raise FeatureJoinError("elo_parameters_mismatch")
        for side in ("home", "away"):
            form = values.get(side + "_form")
            if isinstance(form, Mapping):
                for name in ("points", "goals_for", "goals_against", "matches"):
                    if name in form:
                        values[f"{side}_form_{name}"] = form[name]
                if "rest_days" in form:
                    values[f"{side}_rest_days"] = form["rest_days"]
        projected = {}
        for name in sorted(self.allowed_features.intersection(values)):
            value = values[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
                raise FeatureJoinError("invalid_numeric_feature", name)
            projected[name] = value
        if not projected:
            raise FeatureJoinError("no_allowed_numeric_features")
        excluded = sorted(name for name in values if not _safe_feature(name))
        result = {"match_id": canonical_id, "prediction_cutoff": cutoff,
                  "information_mode": mode, "feature_version": row["feature_version"],
                  "features_available_at": available, **projected}
        for name in ("home_team_id", "away_team_id", "competition_id", "kickoff"):
            if row.get(name) is not None:
                result[name] = _utc(row[name]) if name == "kickoff" else self._id(row[name])
        normalized_hash = _hash({**result, "source_manifest_sha256": expected})
        supplied = original.get("snapshot_sha256")
        if supplied is not None:
            raw = {k: v for k, v in original.items() if k != "snapshot_sha256"}
            if supplied != _hash(raw):
                raise FeatureJoinError("snapshot_content_hash_mismatch")
        result["snapshot_sha256"] = normalized_hash
        result["input_manifest"] = {"source_manifest": manifest, "source_manifest_sha256": expected,
            "feature_snapshot_sha256": normalized_hash, "feature_version": row["feature_version"],
            "excluded_fields": excluded, "id_mapping": []}
        for name in ("match_id", "home_team_id", "away_team_id", "competition_id"):
            if row.get(name) is not None and str(row[name]) in self.aliases:
                result["input_manifest"]["id_mapping"].append({"field": name, "source_id": str(row[name]),
                    **self.aliases[str(row[name])]})
        return result

    def join(self, fixture, *, prediction_cutoff, information_mode, feature_version,
             required_features=(), allow_missing_values=False):
        fixture = asdict(fixture) if is_dataclass(fixture) else fixture
        cutoff = _utc(prediction_cutoff)
        if fixture.get("information_mode") != information_mode:
            raise FeatureJoinError("fixture_information_mode_mismatch")
        if cutoff >= _utc(fixture["kickoff"]):
            raise FeatureJoinError("feature_cutoff_not_prematch")
        key = (self._id(fixture["match_id"]), cutoff, information_mode, feature_version)
        if key in self._conflicts:
            raise FeatureJoinError("conflicting_duplicate_snapshot")
        if key not in self._rows:
            raise FeatureJoinError("missing_exact_feature_snapshot", dict(zip(KEY_FIELDS, map(str, key))))
        row = deepcopy(self._rows[key])
        for name in ("home_team_id", "away_team_id", "competition_id", "kickoff"):
            if name in row and name in fixture:
                target = _utc(fixture[name]) if name == "kickoff" else self._id(fixture[name])
                if row[name] != target:
                    raise FeatureJoinError("feature_fixture_identity_mismatch", name)
        missing_fields = sorted(name for name in required_features if name not in row)
        missing_values = sorted(name for name in required_features if row.get(name) is None and name in row)
        if missing_fields:
            raise FeatureJoinError("missing_required_feature_fields", missing_fields)
        if missing_values and not allow_missing_values:
            raise FeatureJoinError("missing_required_feature_values", missing_values)
        row["match_id"] = str(fixture["match_id"])
        if str(fixture["match_id"]) in self.aliases:
            row["input_manifest"]["requested_id_mapping"] = {
                "source_id": str(fixture["match_id"]), **self.aliases[str(fixture["match_id"])]}
        row["input_manifest"]["missing_values"] = missing_values
        return row

    def assemble_training_rows(self, matches, *, feature_version, lead_hours=24,
                               required_features=(), information_mode=None, allow_missing_values=False):
        if not math.isfinite(float(lead_hours)) or lead_hours <= 0:
            raise ValueError("Prematch lead hours must be positive and finite")
        accepted, exclusions, nulls, seen, conflicts = [], [], Counter(), {}, set()
        for raw in matches:
            match = asdict(raw) if is_dataclass(raw) else dict(raw)
            cutoff = _utc(match["kickoff"]) - timedelta(hours=lead_hours)
            try:
                features = self.join(match, prediction_cutoff=cutoff,
                    information_mode=information_mode or match["information_mode"],
                    feature_version=feature_version, required_features=required_features,
                    allow_missing_values=allow_missing_values)
                ident = match["match_id"]
                identity = _hash({name: match.get(name) for name in (
                    "match_id", "kickoff", "home_team_id", "away_team_id", "competition_id",
                    "home_goals", "away_goals", "result_available_at", "information_mode")})
                if ident in conflicts or (ident in seen and seen[ident] != identity):
                    conflicts.add(ident)
                    accepted = [item for item in accepted if item["match_id"] != ident]
                    raise FeatureJoinError("conflicting_training_match_versions")
                if ident in seen:
                    continue
                seen[ident] = identity
                row = dict(match)
                row.update({k: v for k, v in features.items() if k in self.allowed_features})
                row.update(prediction_cutoff=cutoff, features_available_at=features["features_available_at"],
                    feature_version=feature_version, feature_input_manifest=features["input_manifest"])
                accepted.append(row)
            except FeatureJoinError as exc:
                exclusions.append({"match_id": match["match_id"], "prediction_cutoff": cutoff.isoformat(),
                                   "reason": exc.reason, "detail": exc.detail})
        for row in accepted:
            nulls.update(row["feature_input_manifest"]["missing_values"])
        return accepted, {"accepted_rows": len(accepted), "excluded_rows": len(exclusions),
                          "exclusions": exclusions, "missing_values": dict(nulls),
                          "feature_version": feature_version, "lead_hours": float(lead_hours)}

    def report(self):
        return {"accepted_snapshots": len(self._rows), "identical_duplicates": self.duplicates,
                "conflicting_keys": len(self._conflicts), "rejected_rows": len(self.rejections),
                "rejections": deepcopy(self.rejections), "feature_versions": sorted({k[3] for k in self._rows}),
                "sources": deepcopy(self.sources), "metadata": deepcopy(self.metadata),
                "allowed_features": sorted(self.allowed_features)}


def load_feature_index(path, *, aliases=None, allowed_features=None, strict=False):
    """Read a frozen JSONL, JSON or Parquet file or completed platform bundle.

    Platform history bundles share the history adapter's COMPLETE.json verifier.
    Simple feature-only directories use _SUCCESS and declare
    ``files: {relative_path: sha256}``; every declared file is verified before use.
    A single explicit file is fingerprinted directly and is useful for platform
    exports whose completion is handled by their producer's atomic rename.
    """
    path = Path(path).resolve()
    metadata = None
    if path.is_dir():
        if (path / "COMPLETE.json").is_file():
            from .platform_history import verify_completed_export
            export = verify_completed_export(path)
            if "features" not in export.payloads:
                raise ValueError("Completed platform export has no features file")
            result = FeatureIndex.from_rows(export.rows("features"), aliases=aliases,
                                            allowed_features=allowed_features, strict=strict)
            result.metadata.append(export.manifest)
            result.sources.append({"path": str(path), "manifest_sha256": export.manifest_sha256,
                                   "sha256": sha256(export.payloads["features"]).hexdigest()})
            return result
        if not (path / "_SUCCESS").is_file():
            raise ValueError("Feature export directory is incomplete: missing _SUCCESS")
        metadata_path = path / "manifest.json"
        if not metadata_path.is_file():
            raise ValueError("Feature export directory requires manifest.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = metadata.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Feature export manifest requires files and their SHA256 hashes")
        paths = []
        for relative, expected in sorted(files.items()):
            source = (path / relative).resolve()
            if not source.is_relative_to(path):
                raise ValueError("Feature export path escapes its directory")
            if sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Feature export file hash mismatch: {relative}")
            if source.suffix.lower() in {".jsonl", ".parquet"} and "feature" in source.name:
                paths.append(source)
        if not paths:
            raise ValueError("No declared feature JSONL/Parquet files")
    else:
        paths = [path]
    rows, sources = [], []
    for source in paths:
        digest = sha256(source.read_bytes()).hexdigest()
        sources.append({"path": str(source), "sha256": digest})
        if source.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            rows.extend(pq.read_table(source).to_pylist())
        elif source.suffix.lower() == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
            rows.extend(data if isinstance(data, list) else data.get("snapshots", [data]))
        else:
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    result = FeatureIndex.from_rows(rows, aliases=aliases, allowed_features=allowed_features, strict=strict)
    result.sources = sources
    if metadata is not None:
        result.metadata.append(metadata)
    return result
