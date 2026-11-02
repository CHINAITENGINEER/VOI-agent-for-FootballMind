"""Research-only reconstruction using the platform's parser and rating function.

This is an offline consumer fallback until the platform publishes its completed
export. It neither writes the platform repository nor duplicates provider
parsing or Elo. The output explicitly identifies the research producer.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path

from football_research.baselines.api import canonical_json, content_hash, utc
from football_spider.models.contracts import Observation
from football_spider.normalization.understat import normalize_understat, audit_season, PARSER_VERSION
from football_spider.features.store import compute, PARAMETERS, VERSION
from .platform_history import EXPORT_SCHEMA

FEATURE_VERSION = "research-normalized-history-v1"
ASSUMPTIONS = {"result_delay_hours": 3, "xg_delay_hours": 24, "fixture_lead_hours": 168,
    "kickoff_timezone_policy": "UTC interpretation following platform normalizer and archived official frontend timezone conversion; no independent fixture-time verification",
    "rolling_xg_window": 5}


def feature_rows(matches, statistics, *, source_evidence, assumptions, leads=(24, 1)):
    """Select known normalized inputs, reuse platform Elo, lag normalized xG.

    Result and xG gates are separate. The target record passed to compute has no
    score; all history passed to that function already satisfies result delay.
    xG means use up to five most recent *available* values in this competition.
    """
    for name in ("result_delay_hours", "xg_delay_hours", "fixture_lead_hours"):
        value = assumptions[name]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid reconstruction delay")
    if assumptions["result_delay_hours"] < 2 or assumptions["xg_delay_hours"] < assumptions["result_delay_hours"]:
        raise ValueError("Result/xG delay order invalid")
    if not leads or any(not math.isfinite(lead) or lead <= 0 or lead > assumptions["fixture_lead_hours"] for lead in leads):
        raise ValueError("Lead requires a positive, assumed-known prematch fixture")
    window = assumptions["rolling_xg_window"]
    if type(window) is not int or window < 1:
        raise ValueError("Rolling window must be a positive integer")
    stat_index = {}
    for stat in statistics:
        key = (stat.source_match_id, stat.source_team_id, stat.metric)
        if key in stat_index:
            raise ValueError("Duplicate normalized statistic identity")
        stat_index[key] = stat.value
    history = []
    for match in sorted(matches, key=lambda m: (m.kickoff, m.source_id)):
        history.append({"id": content_hash(match.model_dump(mode="json")),
            "match_id": "understat:match:" + match.source_id,
            "source_id": match.source_id, "competition_id": "understat:competition:" + match.competition.source_id,
            "home_id": "understat:team:" + match.home.source_id,
            "away_id": "understat:team:" + match.away.source_id,
            "event_time": match.kickoff, "record": match.model_dump(mode="json"),
            "result_available_at": match.kickoff + timedelta(hours=assumptions["result_delay_hours"]),
            "xg_available_at": match.kickoff + timedelta(hours=assumptions["xg_delay_hours"])})
    if len({r["match_id"] for r in history}) != len(history):
        raise ValueError("Duplicate normalized match identity")
    rows = []
    for target in history:
        for lead in leads:
            cutoff = target["event_time"] - timedelta(hours=lead)
            eligible = [row for row in history if row["match_id"] != target["match_id"]
                and row["competition_id"] == target["competition_id"] and row["result_available_at"] <= cutoff
                and row["record"]["status"] == "finished" and row["record"]["score_period"] == "regulation"
                and None not in row["record"]["score"].values()]
            # Only identity and kickoff go to the platform feature calculation.
            fixture = {k: target[k] for k in ("match_id", "home_id", "away_id", "event_time")}
            values, version_ids, _ = compute(fixture, eligible, [], cutoff)
            features = {"home_elo": values["elo"]["home"], "away_elo": values["elo"]["away"]}
            xg_inputs = set()
            for side in ("home", "away"):
                team_id = target[side + "_id"].split(":")[-1]
                for metric, field in (("xg", "for"), ("xga", "against")):
                    candidates = [r for r in eligible if r["xg_available_at"] <= cutoff
                        and (r["source_id"], team_id, metric) in stat_index][-window:]
                    features[f"{side}_rolling_xg_{field}"] = (
                        sum(stat_index[(r["source_id"], team_id, metric)] for r in candidates) / len(candidates)
                        if candidates else None)
                    xg_inputs.update(r["match_id"] for r in candidates)
            inputs = [{"match_id": row["match_id"], "version_id": row["id"],
                       "kickoff": row["event_time"], "result_available_at": row["result_available_at"],
                       **({"xg_available_at": row["xg_available_at"]} if row["match_id"] in xg_inputs else {})}
                      for row in eligible]
            upper = max([target["event_time"] - timedelta(hours=assumptions["fixture_lead_hours"])] +
                        [r["result_available_at"] for r in eligible] +
                        [r["xg_available_at"] for r in eligible if r["match_id"] in xg_inputs])
            identity = {"match_id": target["match_id"], "prediction_cutoff": cutoff,
                        "information_mode": "historical_reconstruction", "feature_version": FEATURE_VERSION}
            manifest = {**identity, "source_evidence": source_evidence, "assumptions": assumptions,
                "parameters": PARAMETERS, "rating_implementation": "football_spider.features.store.compute",
                "platform_feature_version": VERSION, "history_inputs": inputs,
                "history_version_ids": version_ids, "xg_history_match_ids": sorted(xg_inputs),
                "max_input_available_at": upper, "fixture_availability_is_assumed": True,
                "missing_value_policy": "null when no prior eligible history; supervised preprocessing fitted only on training rows"}
            rows.append({**identity, "kickoff": target["event_time"], "home_team_id": target["home_id"],
                "away_team_id": target["away_id"], "competition_id": target["competition_id"],
                "features_available_at": upper, "features": features,
                "input_manifest": manifest, "input_manifest_hash": content_hash(manifest)})
    return rows


def build_bundle(manifest_path, output_dir, *, workspace_root="."):
    root = Path(workspace_root).resolve()
    source = json.loads(Path(manifest_path).read_bytes())
    payload_path = (root / source["payload"]).resolve()
    if not payload_path.is_relative_to(root / "var/parallel/collectors"):
        raise ValueError("Use the real collector archive, not fixtures")
    raw = payload_path.read_bytes()
    if sha256(raw).hexdigest() != source["sha256"] or source["http_status"] != 200:
        raise ValueError("Source integrity/status mismatch")
    if source["url"] != "https://understat.com/getLeagueData/EPL/2025":
        raise ValueError("Only the verified EPL/2025 source is supported")
    obs = Observation(id=source["evidence_id"], source="understat", endpoint=source["url"],
        observed_at=utc(source["observed_at"]), payload_path=str(payload_path), content_hash=source["sha256"],
        payload_format="json", http_status=200,
        context={"league": "EPL", "season": 2025, "kickoff_timezone": "UTC", "timezone_policy": ASSUMPTIONS["kickoff_timezone_policy"]})
    # JSON decoding is generic; all provider field interpretation stays in the platform.
    decoded = json.loads(raw)
    batch = normalize_understat(obs, decoded)
    coverage = audit_season(decoded, "EPL", 2025)
    if any(issue.severity == "error" for issue in batch.issues) or not coverage["structural_complete"]:
        raise ValueError("Platform normalization/coverage audit failed")
    evidence = [{"observation_id": obs.id, "sha256": source["sha256"],
                 "observed_at": source["observed_at"], "path": str(payload_path)}]
    matches = [{"match_id": "understat:match:" + m.source_id,
        "competition_id": "understat:competition:" + m.competition.source_id, "season": m.season,
        "home_team_id": "understat:team:" + m.home.source_id, "away_team_id": "understat:team:" + m.away.source_id,
        "home_team_name": m.home.name, "away_team_name": m.away.name, "competition_name": m.competition.name,
        "kickoff": m.kickoff, "status": m.status, "score_period": m.score_period,
        "home_goals": m.score.home, "away_goals": m.score.away,
        "input_manifest": ["normalized_match:sha256:" + content_hash(m.model_dump(mode="json"))]}
        for m in sorted(batch.matches, key=lambda m: (m.kickoff, m.source_id))]
    features = feature_rows(batch.matches, batch.statistics, source_evidence=evidence, assumptions=ASSUMPTIONS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for key, rows in (("matches", matches), ("features", features)):
        data = ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")
        (output / f"{key}.jsonl").write_bytes(data)
        files[key] = {"path": f"{key}.jsonl", "sha256": sha256(data).hexdigest(), "row_count": len(rows)}
    versions = {name: sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()
                for name, fn in (("platform_normalizer", normalize_understat), ("platform_features", compute), ("research_adapter", build_bundle))}
    manifest = {"schema": EXPORT_SCHEMA, "producer": "football_research.data_adapters.reconstruction_bundle",
        "production_platform_export": False, "dataset_version": "research-understat-2025:" + content_hash([files, versions]),
        "information_mode": "historical_reconstruction", "source": "understat",
        "source_observed_at": source["observed_at"], "source_evidence": evidence, "assumptions": ASSUMPTIONS,
        "source_revision_bias": "September 2026 snapshot of 2025/26 season may contain retrospective corrections; kickoff revisions/postponement announcements unavailable; seven-day fixture visibility is assumed",
        "normalizer_version": PARSER_VERSION, "feature_version": FEATURE_VERSION, "implementation_sha256": versions,
        "coverage_audit": coverage, "normalization_issues": [i.model_dump(mode="json") for i in batch.issues], "files": files}
    data = (canonical_json(manifest) + "\n").encode("utf-8")
    (output / "manifest.json").write_bytes(data)
    (output / "COMPLETE.json").write_text(canonical_json({"manifest_sha256": sha256(data).hexdigest()}) + "\n", encoding="utf-8")
    return {"matches": len(matches), "features": len(features), "feature_version": FEATURE_VERSION,
            "manifest_sha256": sha256(data).hexdigest(), "output_dir": str(output.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(canonical_json(build_bundle(args.source_manifest, args.output)))


if __name__ == "__main__":
    main()
