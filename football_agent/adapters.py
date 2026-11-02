"""Read-only operational archive tools and the real research prediction service.

This module owns Agent projections only. It reuses the existing Observation,
RawArchive and normalizers; it neither collects external data nor manufactures
probabilities. Source-scoped IDs remain explicit until canonical identity exists.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from .types import ToolResult
from .model_catalog import ModelCatalog, required_inputs


FEATURE_VERSION = "agent-market-input-v2"
MODEL_INPUT_VERSION = "agent-model-input-v2"
SUPPORTED_TOOLS = ("query_matches", "read_evidence", "get_features", "predict", "refresh_information")


def utc(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(body.encode("utf-8")).hexdigest()


def _platform():
    # Platform owns package imports; this adapter no longer changes sys.path.
    from football_spider.storage.raw_archive import RawArchive
    from football_spider.normalization.sources import normalize
    from football_spider.models.contracts import Observation
    return RawArchive, normalize, Observation


class LocalArchiveTools:
    mode = "real"

    def __init__(self, archive_root="football_spider/platform_data/raw", model_artifact=None,
                 refresh_service=None, *, model_id="market", model_catalog=None,
                 information_mode="operational"):
        self.archive_root = Path(archive_root).resolve()
        if information_mode not in {"operational", "historical_reconstruction"}:
            raise ValueError("Unsupported information mode")
        self.information_mode = information_mode
        self.model_catalog = model_catalog or ModelCatalog()
        self.model_id = model_id
        # Legacy artifact injection is a trusted Python-service-only interface.
        # HTTP/CLI callers select a registered ID; they never supply this path.
        self._trusted_artifact = Path(model_artifact).resolve() if model_artifact else None
        if self._trusted_artifact:
            self.model_artifact = self._trusted_artifact
            self.model_reference = {"model_id": "trusted-service-artifact", "model_name": "market",
                                    "model_artifact": str(self._trusted_artifact), "artifact_sha256": None,
                                    "information_mode": information_mode, "required_inputs": required_inputs("market")}
            try:
                from football_research.baselines.api import PredictionService
                service = PredictionService.load(self._trusted_artifact)
                artifact = service.to_artifact()
                self.model_reference.update(model_name=service.model_name, artifact_sha256=artifact["artifact_sha256"],
                                            model_version=artifact["model_state"]["model_version"],
                                            trained_at=service.trained_at.isoformat(), information_mode=service.information_mode,
                                            required_inputs=required_inputs(service.model_name, service.model))
            except (ImportError, OSError, ValueError, KeyError):
                pass  # Preserve the existing explicit MODEL_UNAVAILABLE tool result.
        else:
            self.model_reference = self.model_catalog.resolve(model_id)
            artifact = self.model_reference.get("model_artifact")
            self.model_artifact = Path(artifact) if artifact else None
            if self.model_artifact is None and self.model_reference["model_name"] == "market":
                self.model_reference["information_mode"] = self.information_mode
        self.refresh_service = refresh_service
        self.archive_issues: list[dict] = []

    def _check_model_reference(self):
        from football_research.baselines.api import PredictionService
        if self._trusted_artifact:
            service = PredictionService.load(self._trusted_artifact)
            actual = service.to_artifact()["artifact_sha256"]
            expected = self.model_reference.get("artifact_sha256")
            if expected and actual != expected:
                raise ValueError("Configured model changed after adapter initialization")
        else:
            current = self.model_catalog.resolve(self.model_id)
            if current["artifact_sha256"] != self.model_reference["artifact_sha256"]:
                raise ValueError("Registered model changed after adapter initialization")

    def capabilities(self) -> dict:
        model_status = "OK"
        model_error = None
        try:
            self._check_model_reference()
        except (ImportError, OSError, ValueError, KeyError) as exc:
            model_status, model_error = "MODEL_UNAVAILABLE", str(exc)
        return {**self.model_reference, "mode": self.mode, "information_mode": self.information_mode,
                "status": model_status, "error": model_error,
                "archive_root": str(self.archive_root),
                "model_artifact": str(self.model_artifact.resolve()) if self.model_artifact else None,
                "availability_basis": "raw_observed_at_or_max_known_ingested_at; current_parser_replay",
                "tools": {name: ("NEEDS_CONFIGURATION" if name == "refresh_information" and self.refresh_service is None
                                  else model_status if name == "predict"
                                  else "OK" if self.archive_root.exists() else "NO_DATA")
                          for name in SUPPORTED_TOOLS},
                "model": "football_research.baselines.api.PredictionService / " + self.model_reference["model_name"],
                "model_error": model_error, "feature_version": (FEATURE_VERSION if self.model_reference["model_name"] == "market" else MODEL_INPUT_VERSION),
                "supported_information_modes": ["operational"],
                "model_registry_path": self.model_catalog.registry_path,
                "entity_ids": "source_scoped_not_canonical", "archive_issues": self.archive_issues,
                "external_requests_enabled": self.refresh_service is not None}

    def estimate_cost(self, tool: str, inputs: dict) -> float:
        if tool == "refresh_information" and self.refresh_service is not None:
            estimator = getattr(self.refresh_service, "estimate_cost", None)
            if estimator is None:
                raise ValueError("Refresh service must expose estimate_cost(inputs) in credits")
            value = float(estimator(inputs))
            if not math.isfinite(value) or value < 0:
                raise ValueError("Refresh service returned an invalid cost estimate")
            return value
        return 0.0

    def execute(self, tool: str, inputs: dict) -> ToolResult:
        if tool not in SUPPORTED_TOOLS:
            return ToolResult(status="FAILED", error=f"Unknown tool: {tool}")
        try:
            return getattr(self, "_" + tool)(dict(inputs))
        except TimeoutError as exc:
            return ToolResult(status="TIMEOUT", error=str(exc) or "Source timed out")
        except (ImportError, ModuleNotFoundError) as exc:
            return ToolResult(status="MODEL_UNAVAILABLE" if tool == "predict" else "NEEDS_CONFIGURATION", error=str(exc))
        except Exception as exc:
            return ToolResult(status="FAILED", error=f"{type(exc).__name__}: {exc}")

    def _observations(self, cutoff: datetime):
        RawArchive, normalize, Observation = _platform()
        archive = RawArchive(self.archive_root.parent)
        self.archive_issues = []
        projected = []
        for path in sorted(self.archive_root.glob("*/*/*.observation.json")):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                # Availability uses actual ingestion when explicitly recorded;
                # file modification dates and SQLite row dates are never used.
                observed = utc(metadata["observed_at"])
                context = metadata.get("context") or {}
                ingested = metadata.get("ingested_at") or context.get("ingested_at")
                available = max(observed, utc(ingested)) if ingested else observed
                if available > cutoff:
                    continue
                obs = Observation.model_validate({k: v for k, v in metadata.items() if k in Observation.model_fields})
                if obs.source not in {"spftll", "football_data", "odds_api"}:
                    continue
                if obs.error_class or obs.http_status is None or not 200 <= obs.http_status < 300:
                    failure_status = ("RATE_LIMITED" if obs.http_status == 429 else
                                      "TIMEOUT" if "timeout" in str(obs.error_class).lower() else "FAILED")
                    self.archive_issues.append({"raw_ref": str(path), "status": failure_status,
                                                "http_status": obs.http_status, "error": obs.error_class})
                    continue
                batch = normalize(obs, json.loads(archive.read_body(obs)))
                projected.append((obs, batch, available, str(path)))
                if batch.issues:
                    self.archive_issues.extend({"raw_ref": str(path), "status": "QUALITY_ISSUE", **i.model_dump(mode="json")}
                                               for i in batch.issues)
            except Exception as exc:
                self.archive_issues.append({"raw_ref": str(path), "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        return sorted(projected, key=lambda item: (item[2], item[0].id))

    def _empty_status(self):
        for issue in self.archive_issues:
            if issue.get("status") in {"TIMEOUT", "RATE_LIMITED", "FAILED"}:
                return issue["status"]
            if issue.get("severity") == "error":
                return "FAILED"
        return "NO_DATA"

    @staticmethod
    def _fixture(record, obs, available, raw_ref):
        source = obs.source
        return {"match_id": f"{source}:{record.source_id}", "source_match_id": record.source_id,
                "source": source, "home": record.home.name, "away": record.away.name,
                "competition": record.competition.name, "competition_id": f"{source}:competition:{record.competition.source_id}",
                "home_team_id": f"{source}:team:{record.home.source_id}",
                "away_team_id": f"{source}:team:{record.away.source_id}",
                "kickoff": record.kickoff.isoformat(), "status": record.status, "score_period": record.score_period,
                "fixture_available_at": available.isoformat(), "information_mode": "operational",
                "identity_quality": {"home": record.home.id_quality, "away": record.away.id_quality},
                "input_manifest": [raw_ref, obs.content_hash], "raw_observation_id": obs.id}

    def _query_matches(self, inputs):
        cutoff = utc(inputs.get("prediction_cutoff") or inputs.get("cutoff") or datetime.now(timezone.utc))
        matches = {}
        for obs, batch, available, raw_ref in self._observations(cutoff):
            for record in batch.matches:
                if record.source_published_at and utc(record.source_published_at) > cutoff:
                    continue
                item = self._fixture(record, obs, available, raw_ref)
                if inputs.get("match_id") and item["match_id"] != inputs["match_id"]:
                    continue
                matches[item["match_id"]] = item
        result = sorted(matches.values(), key=lambda item: (item["kickoff"], item["match_id"]))
        return ToolResult(status="OK" if result else self._empty_status(), data=result,
                          error=None if result else {"reason": "No readable fixture observed by requested cutoff", "archive_issues": self.archive_issues})

    @staticmethod
    def _evidence(obs, available, raw_ref, match_id, kind, key, content):
        content_hash = digest(content)
        return {"evidence_id": f"{obs.id}:{kind}:{key}", "version": content_hash,
                "match_id": match_id, "source": obs.source, "kind": kind, "topic": kind,
                "observed_at": obs.observed_at.isoformat(), "available_at": available.isoformat(),
                "source_published_at": content.get("source_published_at"), "content_hash": content_hash,
                "content": content, "raw_ref": raw_ref, "raw_payload_ref": obs.payload_path,
                "raw_content_hash": obs.content_hash, "information_mode": "operational",
                "conflict_key": (f"{match_id}:{kind}" if kind == "fixture" else
                    f"{match_id}:{kind}:{content.get('provider')}:{content.get('bookmaker')}:{content.get('market_type')}:{content.get('period')}:{content.get('line')}:{content.get('settlement_rule')}:{content.get('source_timestamp') or available.isoformat()}")}

    def _read_evidence(self, inputs):
        match_id, cutoff = inputs["match_id"], utc(inputs["prediction_cutoff"])
        evidence = []
        for obs, batch, available, raw_ref in self._observations(cutoff):
            for record in batch.matches:
                if record.source_published_at and utc(record.source_published_at) > cutoff:
                    continue
                if f"{obs.source}:{record.source_id}" == match_id:
                    evidence.append(self._evidence(obs, available, raw_ref, match_id, "fixture", record.source_id,
                                                   record.model_dump(mode="json")))
            groups = defaultdict(list)
            for quote in batch.quotes:
                if quote.source_timestamp and utc(quote.source_timestamp) > cutoff:
                    continue
                if f"{obs.source}:{quote.source_match_id}" == match_id:
                    key = (quote.provider, quote.bookmaker, quote.market_type, quote.period,
                           quote.line, quote.source_timestamp, quote.settlement_rule)
                    groups[key].append(quote.model_dump(mode="json"))
            for key, quotes in groups.items():
                content = {"quotes": sorted(quotes, key=lambda quote: quote["selection"]),
                           "provider": key[0], "bookmaker": key[1], "market_type": key[2], "period": key[3],
                           "line": key[4], "source_timestamp": key[5].isoformat() if key[5] else None,
                           "settlement_rule": key[6]}
                evidence.append(self._evidence(obs, available, raw_ref, match_id, "odds", digest(content)[:16], content))
        evidence.sort(key=lambda row: (row["available_at"], row["evidence_id"]))
        return ToolResult(status="OK" if evidence else self._empty_status(), data=evidence,
                          error=None if evidence else {"reason": "No readable evidence observed by requested cutoff", "archive_issues": self.archive_issues})

    def _get_features(self, inputs):
        cutoff = utc(inputs["prediction_cutoff"])
        model_name = self.model_reference["model_name"]
        if model_name in {"frequency", "poisson", "dixon_coles"}:
            match = inputs.get("match")
            if not match:
                return ToolResult(status="NO_DATA", error="Selected model requires a visible fixture")
            error = self._validate_match(inputs, match)
            if error:
                return ToolResult(status="NO_DATA", error=error)
            manifest = list(match.get("input_manifest", []))
            if not manifest:
                return ToolResult(status="NO_DATA", error="Fixture requires input lineage")
            # Execution envelope carries provenance/version while the fitted
            # service consumes fixture fields and its frozen training state.
            features = {"match_id": inputs["match_id"], "prediction_cutoff": cutoff.isoformat(),
                        "feature_version": MODEL_INPUT_VERSION, "information_mode": self.information_mode,
                        "model_id": self.model_id, "model_artifact_sha256": self.model_reference["artifact_sha256"],
                        "input_manifest": manifest, "feature_snapshot_required": False,
                        "required_inputs": self.model_reference["required_inputs"],
                        "features_available_at": match["fixture_available_at"], "missing": {}}
            features["feature_snapshot_hash"] = digest(features)
            return ToolResult(data=features)
        if model_name != "market":
            return ToolResult(status="NEEDS_CONFIGURATION", error={
                "reason": "Selected model requires a platform FeatureSnapshot",
                "model_name": model_name, "required_inputs": self.model_reference["required_inputs"]})
        candidates = []
        inconsistent = []
        for item in inputs.get("evidence", []):
            if item.get("match_id") != inputs["match_id"] or item.get("kind") != "odds":
                continue
            if item.get("information_mode", "operational") != self.information_mode:
                continue
            if utc(item["observed_at"]) > cutoff or utc(item.get("available_at", item["observed_at"])) > cutoff:
                continue
            if item.get("source_published_at") and utc(item["source_published_at"]) > cutoff:
                continue
            if item.get("expires_at") and utc(item["expires_at"]) <= cutoff:
                continue
            content = item["content"]
            if content.get("market_type") != "1x2" or content.get("period") != "regulation" or content.get("line") is not None:
                continue
            quotes = content.get("quotes", [])
            selections = {q["selection"]: q["decimal_price"] for q in quotes}
            if len(quotes) != 3 or set(selections) != {"home", "draw", "away"}:
                continue
            if any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 1 for value in selections.values()):
                continue
            if content.get("source_timestamp") and utc(content["source_timestamp"]) > cutoff:
                continue
            # Never assemble a complete market from different publishers,
            # settlement rules, periods, lines, or quote timestamps.
            dimensions = ("provider", "bookmaker", "market_type", "period", "line", "settlement_rule", "source_timestamp")
            bad_group = False
            for dimension in dimensions:
                supplied = [q[dimension] for q in quotes if dimension in q]
                target = content.get(dimension)
                if dimension == "source_timestamp":
                    supplied = [utc(v).isoformat() if v is not None else None for v in supplied]
                    target = utc(target).isoformat() if target is not None else None
                if supplied and (len({json.dumps(v, sort_keys=True) for v in supplied}) != 1 or
                                 any(v != target for v in supplied)):
                    bad_group = True
            if bad_group:
                inconsistent.append(item["evidence_id"])
                continue
            available = utc(item.get("available_at", item["observed_at"]))
            group = tuple(content.get(key) for key in dimensions[:-1])
            version = (utc(content["source_timestamp"]).isoformat() if content.get("source_timestamp") else
                       item.get("observation_version") or available.isoformat())
            candidates.append((available, item, selections, group, version))
        if not candidates:
            return ToolResult(status="NO_DATA", error={"reason": "No complete visible regulation 1x2 snapshot",
                                                       "rejected_inconsistent_groups": inconsistent})
        by_fact = defaultdict(list)
        for row in candidates:
            by_fact[(row[3], row[4])].append(row)
        conflicts = [group for group in by_fact.values() if len({digest(row[2]) for row in group}) > 1]
        if conflicts:
            return ToolResult(status="FAILED", error={"reason": "Conflicting quotes for the same market fact/version",
                "evidence_ids": [[row[1]["evidence_id"] for row in group] for group in conflicts]})
        # Latest complete observation wins; simultaneous independent publishers
        # use lexicographic provider/bookmaker priority, never price averaging.
        candidates.sort(key=lambda row: (-row[0].timestamp(), str(row[3][0] or ""),
                                          str(row[3][1] or ""),
                                          -utc(row[1]["content"].get("source_timestamp") or row[1]["observed_at"]).timestamp(),
                                          row[1]["evidence_id"]))
        available, selected, prices, _, _ = candidates[0]
        content = selected["content"]
        features = {"match_id": inputs["match_id"], "prediction_cutoff": cutoff.isoformat(),
                    "feature_version": FEATURE_VERSION, "information_mode": self.information_mode,
                    "model_id": self.model_id, "model_artifact_sha256": self.model_reference["artifact_sha256"],
                    "odds_home": prices["home"], "odds_draw": prices["draw"], "odds_away": prices["away"],
                    "odds_available_at": available.isoformat(), "odds_source_timestamp": content.get("source_timestamp"),
                    "odds_period": "regulation", "odds_market_type": "1x2",
                    "odds_snapshot_id": selected["evidence_id"], "provider": content["provider"],
                    "bookmaker": content.get("bookmaker"), "settlement_rule": content["settlement_rule"],
                    "input_manifest": [{"evidence_id": selected["evidence_id"], "version": selected["version"],
                                        "raw_ref": selected["raw_ref"], "raw_content_hash": selected.get("raw_content_hash")}],
                    "selection_rule": "latest_complete_observation_then_provider_bookmaker_lexicographic",
                    "candidate_evidence_ids": [row[1]["evidence_id"] for row in candidates],
                    "independent_market_groups": len({row[3] for row in candidates}),
                    "required_inputs": self.model_reference["required_inputs"], "missing": {}}
        features["feature_snapshot_hash"] = digest(features)
        return ToolResult(data=features)

    def _validate_match(self, inputs, match):
        if str(match.get("match_id")) != str(inputs["match_id"]):
            return "Fixture belongs to another match"
        if match.get("information_mode") != self.information_mode:
            return "Fixture information mode differs from selected run mode"
        if self.model_reference.get("information_mode") != self.information_mode:
            return "Selected model information mode differs from run mode"
        competition = self.model_reference.get("competition_id")
        if competition and match.get("competition_id") != competition:
            return "Selected trained model is registered for a different competition"
        if not match.get("fixture_available_at") or utc(match["fixture_available_at"]) > utc(inputs["prediction_cutoff"]):
            return "Fixture is unavailable at prediction cutoff"
        return None

    def _predict(self, inputs):
        from football_research.baselines.api import PredictionService
        if not inputs.get("match"):
            return ToolResult(status="NO_DATA", error="Visible fixture is required")
        error = self._validate_match(inputs, inputs["match"])
        if error:
            return ToolResult(status="NO_DATA", error=error)
        features = inputs.get("features")
        needs_snapshot = self.model_reference["required_inputs"]["feature_snapshot_required"]
        if needs_snapshot and not features:
            return ToolResult(status="NO_DATA", error="Selected model requires a visible feature snapshot")
        try:
            self._check_model_reference()
        except (ImportError, OSError, ValueError, KeyError) as exc:
            return ToolResult(status="MODEL_UNAVAILABLE", error=str(exc))
        try:
            if features:
                if str(features.get("match_id")) != str(inputs["match_id"]):
                    raise ValueError("Features belong to another match")
                if utc(features["prediction_cutoff"]) != utc(inputs["prediction_cutoff"]):
                    raise ValueError("Features must share the exact prediction cutoff")
                if features.get("information_mode") != self.information_mode:
                    raise ValueError("Feature information mode differs")
                if not features.get("feature_version") or not features.get("input_manifest"):
                    raise ValueError("Features require a version and input lineage")
                if features.get("model_id", self.model_id) != self.model_id:
                    raise ValueError("Features were prepared for another selected model")
                if features.get("model_artifact_sha256", self.model_reference["artifact_sha256"]) != self.model_reference["artifact_sha256"]:
                    raise ValueError("Features refer to a different model artifact")
                if features.get("feature_snapshot_hash") and digest({k: v for k, v in features.items() if k != "feature_snapshot_hash"}) != features["feature_snapshot_hash"]:
                    raise ValueError("Feature envelope hash mismatch")
            if self.model_artifact is not None:
                if not self.model_artifact.exists():
                    return ToolResult(status="MODEL_UNAVAILABLE", error="Configured model artifact does not exist")
                service = PredictionService.load(self.model_artifact)
            else:
                # The market model is stateless; train([]) freezes its documented
                # transformation and provenance, not fabricated training outcomes.
                service = PredictionService.train("market", [], as_of=inputs["prediction_cutoff"],
                            dataset_version="raw-archive:" + features["feature_snapshot_hash"],
                            information_mode=self.information_mode)
            prediction = service.predict(inputs["match"], prediction_cutoff=inputs["prediction_cutoff"],
                                         features=features if needs_snapshot else None)
            return ToolResult(data=prediction.model_dump(mode="json"))
        except (ImportError, FileNotFoundError) as exc:
            return ToolResult(status="MODEL_UNAVAILABLE", error=str(exc))
        except ValueError as exc:
            return ToolResult(status="NO_DATA", error=str(exc))

    def _refresh_information(self, inputs):
        if self.refresh_service is None:
            return ToolResult(status="NEEDS_CONFIGURATION", data={"evidence": []},
                              error="Existing collection service is not connected; local replay performs no network requests")
        result = self.refresh_service.execute(inputs)
        if not isinstance(result, ToolResult):
            raise TypeError("Refresh service must return football_agent.types.ToolResult")
        if result.status != "OK":
            return result
        # A successful new collection does not make today's evidence visible at a
        # historical cutoff. Re-read the immutable archive through the same gate.
        evidence = self._read_evidence(inputs)
        return ToolResult(status=evidence.status, data={"evidence": evidence.data or [], "refresh": result.data},
                          error=evidence.error, cost=result.cost, retry_after_seconds=result.retry_after_seconds,
                          requests=result.requests)


class ArchiveReplayTools(LocalArchiveTools):
    """Replay real archived arrivals, without issuing new collection requests.

    Initial read reveals the first recorded observation. A topic query reveals
    later archived observations for that topic, all still bounded by cutoff.
    This deterministic disclosure is explicitly identified as archive replay;
    the observation times remain the original source retrieval times.
    """

    def capabilities(self):
        result = super().capabilities()
        result["tools"]["refresh_information"] = "OK"
        result["refresh_mechanism"] = "archive_replay"
        result["label"] = "真实档案回放 / 本地逐批披露"
        result["external_requests_enabled"] = False
        return result

    def estimate_cost(self, tool, inputs):
        return 0.0

    def _read_evidence(self, inputs):
        result = super()._read_evidence(inputs)
        if result.status != "OK":
            return result
        first_at = min(item["available_at"] for item in result.data)
        return ToolResult(data=[item for item in result.data if item["available_at"] == first_at])

    def _refresh_information(self, inputs):
        result = super()._read_evidence(inputs)
        if result.status != "OK":
            return ToolResult(status=result.status, data={"evidence": [], "mechanism": "archive_replay"}, error=result.error)
        first_at = min(item["available_at"] for item in result.data)
        topic = inputs.get("topic")
        evidence = [item for item in result.data if item["available_at"] > first_at and item["topic"] == topic]
        return ToolResult(status="OK" if evidence else "NO_DATA",
                          data={"evidence": evidence, "mechanism": "archive_replay", "network_requests": 0},
                          error=None if evidence else "No later archived observations for requested topic by cutoff")
