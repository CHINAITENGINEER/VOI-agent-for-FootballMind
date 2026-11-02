"""Read existing real FeatureStore Parquet through an evidenced ID binding.

The binding is Agent configuration, not a domain contract or new platform
export. Its ID evidence is the platform's actual PostgreSQL acceptance report.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .adapters import LocalArchiveTools, digest, utc
from .platform_tools import PlatformTools, jsonable
from .types import ToolResult


class FrozenFeatureTools(PlatformTools):
    def __init__(self, *, export_path, **kwargs):
        if kwargs.get("time_mode", "archive_replay") != "archive_replay" or kwargs.get("service"):
            raise ValueError("Frozen FeatureStore binding requires archive_replay without refresh")
        super().__init__(export_path=None, **kwargs)
        self.export_path = str(Path(export_path).resolve())
        self.binding_bytes = Path(self.export_path).read_bytes()
        self.binding = json.loads(self.binding_bytes)
        if self.binding.get("schema") != "agent-frozen-platform-binding-v1":
            raise ValueError("Unknown frozen platform binding schema")
        self._load_snapshot()

    def _checked_path(self, key):
        entry = self.binding[key]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = Path(self.export_path).parent / path
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen platform reference hash changed: " + key)
        return path

    def _load_snapshot(self):
        from football_research.data_adapters.features import load_feature_snapshot
        report = json.loads(self._checked_path("platform_report").read_text(encoding="utf-8"))
        self.snapshot = load_feature_snapshot(self._checked_path("feature_snapshot"))
        target = report["target"]
        self.target_id = target["match_id"]
        self.source_id = "spftll:" + target["source_id"]
        if report.get("single_source_acceptance") != "OK" or report.get("fixtures_used") is not False:
            raise ValueError("Binding requires a successful real platform acceptance report")
        if self.snapshot["match_id"] != self.target_id or self.snapshot["input_manifest_hash"] != report["feature"]["manifest_hash"]:
            raise ValueError("Feature identity/manifest differs from platform report")
        if self.snapshot["information_mode"] != self.information_mode:
            raise ValueError("Frozen feature information mode differs")
        self.target = target

    def _verify_export(self):
        if not hasattr(self, "binding"):
            return
        if Path(self.export_path).read_bytes() != self.binding_bytes:
            raise ValueError("Frozen binding changed")
        self._load_snapshot()

    def capabilities(self):
        caps = super().capabilities()
        if not hasattr(self, "binding"):
            return caps
        caps.update(reader_version="platform-featurestore-parquet-binding-v1",
                    export_path=self.export_path, export_sha256=sha256(self.binding_bytes).hexdigest(),
                    expected_feature_version=self.snapshot["feature_version"],
                    feature_version=self.snapshot["feature_version"],
                    feature_snapshot_sha256=self.snapshot["parquet_sha256"],
                    entity_ids="platform_report_evidenced_source_to_canonical_match_binding",
                    identity_scope="single_source_platform_mapping;cross_source_confirmation_not_claimed")
        caps["configuration_hash"] = digest({k: caps[k] for k in (
            "export_path", "export_sha256", "feature_snapshot_sha256", "reader_version", "information_mode")})
        return caps

    def _query_matches(self, inputs):
        if inputs.get("match_id", self.target_id) != self.target_id:
            return ToolResult("NO_DATA", data=[], error="Unknown canonical match ID")
        cutoff = inputs.get("prediction_cutoff") or self.snapshot["prediction_cutoff"]
        if utc(cutoff) != self.snapshot["prediction_cutoff"]:
            return ToolResult("NO_DATA", data=[], error="Frozen snapshot requires its exact cutoff")
        result = LocalArchiveTools._query_matches(self, {"match_id": self.source_id, "prediction_cutoff": cutoff})
        if result.status != "OK":
            return result
        rows = []
        for item in result.data:
            if item["match_id"] != self.source_id:
                continue
            if utc(item["kickoff"]) != utc(self.target["kickoff"]):
                return ToolResult("FAILED", error="Fixture kickoff differs from platform ID evidence")
            item = deepcopy(item)
            item["match_id"] = self.target_id
            item["input_manifest"].extend(["platform_report:sha256:" + self.binding["platform_report"]["sha256"],
                                           "source_match_id:" + self.source_id])
            rows.append(item)
        return ToolResult("OK" if rows else "NO_DATA", data=rows)

    def _read_evidence(self, inputs):
        fixtures = self._query_matches(inputs)
        if fixtures.status != "OK":
            return fixtures
        match = fixtures.data[0]
        return ToolResult(data=[{"evidence_id": "frozen-fixture:" + self.target_id,
            "version": self.snapshot["input_manifest_hash"], "match_id": self.target_id,
            "source": "spftll", "kind": "fixture", "topic": "fixture", "information_mode": self.information_mode,
            "observed_at": match["fixture_available_at"], "available_at": match["fixture_available_at"],
            "content": match, "content_hash": digest(match), "raw_ref": match["input_manifest"][0],
            "feature_manifest": self.snapshot["manifest"], "mapping_evidence": self.binding["platform_report"]}])

    def _get_features(self, inputs):
        error = self._validate_match(inputs, inputs.get("match") or {})
        if error:
            return ToolResult("NO_DATA", error=error)
        if inputs["match_id"] != self.target_id or utc(inputs["prediction_cutoff"]) != self.snapshot["prediction_cutoff"]:
            return ToolResult("NO_DATA", error="Feature identity/cutoff mismatch")
        needs = self.model_reference["required_inputs"]
        if needs["odds_required"]:
            return ToolResult("NO_DATA", error="This frozen projection supplies Elo, not raw 1x2 quotes")
        values = jsonable(self.snapshot)
        if any(values.get(k) is None for k in needs["features"]):
            return ToolResult("NO_DATA", error="Required feature fields missing from snapshot")
        values.update(model_id=self.model_id, model_artifact_sha256=self.model_reference["artifact_sha256"])
        values["feature_snapshot_hash"] = digest(values)
        return ToolResult(data=values)


def platform_tools(**options):
    path = options.get("export_path")
    if path and Path(path).is_file():
        return FrozenFeatureTools(**options)
    return PlatformTools(**options)
