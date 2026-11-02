"""Agent projections of completed platform exports and platform services.

Source parsing, entity resolution, feature computation and collection belong to
the platform. This boundary verifies their results before PredictionService is
called and keeps operational reads separate from historical reconstruction.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from .adapters import LocalArchiveTools, digest, utc
from .types import ToolResult


def jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class PlatformTools(LocalArchiveTools):
    """Trusted server construction; clients select model IDs, never paths.

    Completed history exports use the research package's existing verified
    reader and exact FeatureIndex join. A supplied platform service is an
    operator-owned dependency, not an object constructed from client input.
    """

    def __init__(self, *, model_id="market", model_catalog=None,
                 information_mode="operational", time_mode="archive_replay",
                 export_path=None, service=None,
                 archive_root="football_spider/platform_data/raw"):
        super().__init__(archive_root, model_id=model_id, model_catalog=model_catalog,
                         information_mode=information_mode)
        if time_mode not in {"archive_replay", "live"}:
            raise ValueError("Unsupported time mode")
        if time_mode == "live" and information_mode != "operational":
            raise ValueError("Live decisions require operational information mode")
        self.time_mode = time_mode
        self.export_path = str(Path(export_path).resolve()) if export_path else None
        self.service = service
        self.export = self.dataset = self.feature_index = None
        if self.export_path:
            from football_research.data_adapters.platform_history import (
                verify_completed_export, load_completed_history_export)
            from football_research.data_adapters.batch_features import FeatureIndex
            self.export = verify_completed_export(self.export_path)
            self.dataset = load_completed_history_export(self.export_path)
            if self.dataset.audit["manifest_sha256"] != self.export.manifest_sha256:
                raise ValueError("Platform export changed while loading its dataset")
            if self.export.manifest["information_mode"] != information_mode:
                raise ValueError("Export information mode differs from run mode")
            if "features" in self.export.payloads:
                self.feature_index = FeatureIndex.from_rows(self.export.rows("features"), strict=True)
        self._configuration = self._configuration_reference()

    def _configuration_reference(self):
        return {"export_path": self.export_path,
                "export_sha256": self.export.manifest_sha256 if self.export else None,
                "reader_version": "verified-platform-history-export-v1" if self.export else "platform-raw-observation-v1",
                "time_mode": self.time_mode, "information_mode": self.information_mode,
                "service": self.service.configuration() if self.service is not None else None}

    def capabilities(self):
        caps = super().capabilities()
        reference = self._configuration_reference()
        caps.update(reference)
        caps["configuration_hash"] = digest(reference)
        caps["refresh_mechanism"] = "platform_refresh" if self.service else "platform_refresh_unconfigured"
        caps["external_requests_enabled"] = self.service is not None
        caps["tools"]["refresh_information"] = "OK" if self.service else "NEEDS_CONFIGURATION"
        if self.export:
            caps.update(dataset_version=self.export.manifest["dataset_version"],
                        entity_ids="platform_export_ids_preserved_without_name_join",
                        source_observed_at=self.export.manifest["source_observed_at"],
                        supported_information_modes=[self.information_mode])
            if self.information_mode == "historical_reconstruction":
                caps["reconstruction_assumptions"] = {
                    "basis": self.export.manifest["assumptions"],
                    "revision_risk": self.dataset.audit["source_revision_bias"]}
            for tool in ("query_matches", "read_evidence", "get_features"):
                caps["tools"][tool] = "OK"
        caps["label"] = ("真实历史重建 · 明示延迟与事后修订风险" if self.information_mode == "historical_reconstruction"
                         else "真实在线决策 · 刷新关联新时点" if self.time_mode == "live"
                         else "真实平台读取 · 固定历史截止点")
        return caps

    def _verify_export(self):
        if self.export:
            from football_research.data_adapters.platform_history import verify_completed_export
            current = verify_completed_export(self.export_path)
            if current.manifest_sha256 != self.export.manifest_sha256:
                raise ValueError("Frozen platform export changed after configuration")

    def execute(self, tool, inputs):
        if inputs.get("information_mode", self.information_mode) != self.information_mode:
            return ToolResult("FAILED", error="INFORMATION_MODE_MISMATCH")
        try:
            self._verify_export()
        except (OSError, ValueError, KeyError) as exc:
            return ToolResult("FAILED", error={"reason": "PLATFORM_EXPORT_INVALID", "detail": str(exc)})
        return super().execute(tool, inputs)

    def _query_matches(self, inputs):
        if self.dataset is None:
            return super()._query_matches(inputs)
        cutoff = utc(inputs.get("prediction_cutoff") or datetime.now().astimezone())
        rows = []
        for match in self.dataset.matches:
            if inputs.get("match_id") and inputs["match_id"] != match.match_id:
                continue
            if match.status not in {"scheduled", "unknown", "finished"}:
                continue
            if match.status == "finished" and self.information_mode == "operational":
                continue
            if match.fixture_available_at is None or match.fixture_available_at > cutoff or cutoff >= match.kickoff:
                continue
            # The export may contain later results. Only prematch fixture fields
            # enter this projection; no target result or target xG is copied.
            rows.append({"match_id": match.match_id, "competition_id": match.competition_id,
                         "home_team_id": match.home_team_id, "away_team_id": match.away_team_id,
                         "home": match.home_team_name, "away": match.away_team_name,
                         "competition": match.competition_name, "kickoff": match.kickoff.isoformat(),
                         "fixture_available_at": match.fixture_available_at.isoformat(),
                         "information_mode": match.information_mode, "source": match.source,
                         "score_period": match.score_period, "status": "scheduled",
                         "input_manifest": list(match.input_manifest),
                         "identity_quality": match.entity_quality, "time_basis": match.time_basis})
        return ToolResult("OK" if rows else "NO_DATA", data=rows,
                          error=None if rows else "No prematch fixture visible at requested cutoff and exact ID")

    def _read_evidence(self, inputs):
        if self.dataset is None:
            return super()._read_evidence(inputs)
        result = self._query_matches(inputs)
        if result.status != "OK":
            return result
        assumptions = self.capabilities().get("reconstruction_assumptions")
        rows = []
        for fixture in result.data:
            row = {"evidence_id": "platform-fixture:" + fixture["match_id"],
                   "version": self.export.manifest_sha256, "match_id": fixture["match_id"],
                   "source": fixture["source"], "kind": "fixture", "topic": "fixture",
                   "information_mode": self.information_mode,
                   "observed_at": self.export.manifest["source_observed_at"],
                   "available_at": fixture["fixture_available_at"],
                   "raw_ref": str(self.export.directory / "manifest.json"),
                   "source_evidence": self.export.manifest["source_evidence"],
                   "content": fixture, "content_hash": digest(fixture)}
            if assumptions:
                row["reconstruction"] = assumptions
            rows.append(row)
        return ToolResult(data=rows)

    def _get_features(self, inputs):
        error = self._validate_match(inputs, inputs.get("match") or {})
        if error:
            return ToolResult("NO_DATA", error=error)
        needs = self.model_reference["required_inputs"]
        if self.feature_index is None or not needs["feature_snapshot_required"] or needs["odds_required"]:
            result = super()._get_features(inputs)
        else:
            versions = self.feature_index.report()["feature_versions"]
            if len(versions) != 1:
                return ToolResult("NO_DATA", error="Export must select one unambiguous feature version")
            try:
                result = ToolResult(data=jsonable(self.feature_index.join(inputs["match"],
                    prediction_cutoff=inputs["prediction_cutoff"], information_mode=self.information_mode,
                    feature_version=versions[0], required_features=needs["features"])))
            except ValueError as exc:
                return ToolResult("NO_DATA", error=str(exc))
        if result.status != "OK" or not self.export:
            return result
        envelope = deepcopy(result.data)
        envelope.pop("feature_snapshot_hash", None)
        envelope.update(platform_dataset_version=self.export.manifest["dataset_version"],
                        platform_export_sha256=self.export.manifest_sha256,
                        model_id=self.model_id, model_artifact_sha256=self.model_reference["artifact_sha256"])
        envelope["feature_snapshot_hash"] = digest(envelope)
        return ToolResult(data=envelope)

    def estimate_requests(self, tool, inputs):
        return self.service.estimate_requests(inputs) if tool == "refresh_information" and self.service else 0

    def estimate_cost(self, tool, inputs):
        return self.service.estimate_cost(inputs) if tool == "refresh_information" and self.service else 0.0

    def _refresh_information(self, inputs):
        if self.service is None:
            return ToolResult("NEEDS_CONFIGURATION", data={"evidence": []},
                              error="Platform refresh service has not been configured")
        return self.service.refresh(inputs, reader=self, time_mode=self.time_mode)
