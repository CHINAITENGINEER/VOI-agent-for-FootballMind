"""Eligibility checks for a predeclared, newly captured temporal period.

This supervisor consumes verified platform exports, never decides which
outcomes to retain, and never fits a predictor. Capture after a local freeze
does not establish prospective collection or externally independent gold.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..acquisition.conservative_joint import ConservativeJointReturnPredictor
from ..acquisition.joint_returns import JointReturnPredictor
from ..acquisition.state import parse_time


def load_frozen_models(root, freeze, *, design_path):
    """Load exactly the original and adapted artifacts named by the freeze."""
    root = Path(root).resolve()
    if sha256(Path(design_path).read_bytes()).hexdigest() != freeze["design_sha256"]:
        raise ValueError("Frozen design hash differs")
    models = []
    for name, cls in (("reference", JointReturnPredictor), ("adapted", ConservativeJointReturnPredictor)):
        relative = Path(freeze[name+"_model_path"])
        path = (root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Frozen model path must remain inside the project")
        if sha256(path.read_bytes()).hexdigest() != freeze[name+"_model_sha256"]:
            raise ValueError("Frozen model artifact hash differs")
        models.append(cls.load(path))
    reference, adapted = models
    if (adapted.model_version != freeze["adapted_model_version"]
            or adapted.evidence_strength != freeze["evidence_strength"]
            or adapted.metadata["evidence_strength_adaptation"]["reference_model_version"] != reference.model_version):
        raise ValueError("Frozen models and adaptation do not agree")
    return reference, adapted


def audit_frozen_period(freeze, manifest, rows, *, manifest_sha256, capture, normalized_matches,
                        competition_id, reference_metadata, adapted_metadata):
    """Require all completed regulation matches, with no quality/outcome filter.

``normalized_matches`` must be the complete verified matches payload, not a
caller-selected subset. ``capture`` describes the archived source bytes;
their digest is independently checked by the caller before this audit.
"""
    if (freeze.get("parameter_retuning_permitted") is not False
            or freeze.get("candidate_generation_complete") is not False
            or freeze.get("historical_reconstruction_only") is not True):
        raise ValueError("Frozen historical evaluation scope changed")
    frozen_at = parse_time(freeze["frozen_at"])
    started, observed = parse_time(capture["request_started_at"]), parse_time(capture["observed_at"])
    if not frozen_at < started <= observed:
        raise ValueError("Source discovery did not follow the policy freeze")
    if (capture.get("http_status") != 200 or str(capture.get("season")) != str(freeze["season"])
            or capture.get("league") != freeze["league"]):
        raise ValueError("Source capture scope differs from frozen period")
    if (manifest.get("source") != freeze["source"] or str(manifest.get("season")) != str(freeze["season"])
            or manifest.get("information_mode") != "historical_reconstruction"
            or parse_time(manifest["source_observed_at"]) != observed):
        raise ValueError("Platform manifest scope or source time differs")
    evidence = manifest.get("source_evidence", [])
    if len(evidence) != 1:
        raise ValueError("This frozen evaluation requires the one declared source snapshot")
    item = evidence[0]
    context = item.get("original_context", {})
    if (item.get("content_hash") != capture["sha256"] or item.get("observation_id") != capture["evidence_id"]
            or parse_time(item["observed_at"]) != observed
            or context.get("league") != freeze["league"] or str(context.get("season")) != str(freeze["season"])):
        raise ValueError("Source provenance differs from frozen period")
    matches = list(normalized_matches)
    if (not matches or len({m["match_id"] for m in matches}) != len(matches)
            or len({m["source_id"] for m in matches}) != len(matches)):
        raise ValueError("Complete normalized match identities must be unique")
    for match in matches:
        if (match.get("source") != freeze["source"] or str(match.get("season")) != str(freeze["season"])
                or match.get("competition_id") != competition_id
                or match.get("observation_id") != capture["evidence_id"]):
            raise ValueError("Normalized match source/season/competition differs")
    completed = {m["match_id"]: m for m in matches if m["status"] == "finished" and m["score_period"] == "regulation"}
    ids = [r["match_id"] for r in rows]
    if not completed or len(set(ids)) != len(ids) or set(ids) != set(completed):
        raise ValueError("Episodes must contain every completed regulation match exactly once")
    trained = set(freeze["forbidden_match_ids"])
    for meta in (reference_metadata, adapted_metadata):
        trained.update(meta.get("training_match_ids", []))
        trained.update(meta.get("base_training_match_ids", []))
        trained.update(meta.get("evidence_strength_adaptation", {}).get("selection_match_ids", []))
    if trained.intersection(ids):
        raise ValueError("New-period matches overlap training, adaptation or previously explored IDs")
    earliest = max(parse_time(freeze["minimum_prediction_cutoff"]),
                   parse_time(reference_metadata["training_as_of"]), parse_time(adapted_metadata["training_as_of"]))
    from ..data_adapters.platform_history import ADAPTER_VERSION
    versions = {r.get("dataset_version") for r in rows}
    if versions != {ADAPTER_VERSION+":"+manifest_sha256}:
        raise ValueError("Episode dataset version is not bound to the verified manifest")
    for row in rows:
        original = completed[row["match_id"]]
        prediction, kickoff = parse_time(row["prediction_cutoff"]), parse_time(row["kickoff"])
        result_at = parse_time(row["result_available_at"])
        if not earliest <= prediction < kickoff < result_at <= observed:
            raise ValueError("Episode prediction/label chronology exceeds frozen period or capture")
        if row.get("information_mode") != "historical_reconstruction" or row.get("score_period") != "regulation":
            raise ValueError("Episode information mode or score period differs")
        for field in ("fixture_available_at", "features_available_at"):
            if not row.get(field) or parse_time(row[field]) > prediction:
                raise ValueError("Episode input is unavailable at its prediction cutoff")
        for field in ("competition_id", "home_team_id", "away_team_id", "source", "season", "home_goals", "away_goals"):
            if row.get(field) != original.get(field):
                raise ValueError("Episode differs from the unfiltered normalized source match")
        if kickoff != parse_time(original["kickoff"]) or result_at != parse_time(original["result_available_at"]):
            raise ValueError("Episode match clocks differ from normalized source")
        for field in ("home_goals", "away_goals"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("Completed source requires nonnegative integer scores")
    return {"schema": "frozen-temporal-period-audit-v1", "status": "eligible_historical_temporal_period",
            "matches": len(rows), "normalized_matches": len(matches), "match_ids": sorted(ids),
            "frozen_at": freeze["frozen_at"], "source_request_started_at": capture["request_started_at"],
            "source_observed_at": capture["observed_at"], "all_completed_matches_retained": True,
            "unknown_identity_or_missing_history_filter_applied": False, "prior_match_overlap": 0,
            "minimum_prediction_cutoff": min(parse_time(r["prediction_cutoff"]) for r in rows).isoformat(),
            "maximum_label_available_at": max(parse_time(r["result_available_at"]) for r in rows).isoformat(),
            "prospective_collection": False, "independent_outcome_verified": False,
            "parameter_retuning_permitted": False}
