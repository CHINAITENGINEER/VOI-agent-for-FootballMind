"""Bind an already archived official preview to existing spftll fixture evidence.

No network and no fabricated publication time. The real case demonstrates the
availability gate, not predictive improvement or an operational historical run.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from football_spider.models.contracts import Observation
from football_spider.normalization.sources import stamp
from football_spider.storage.raw_archive import RawArchive, json_bytes
from .audit_round7 import _walk
from .contracts import Fixture
from .evidence_registry import CompetitionEvidenceRegistry


def prepare(workspace: Path, capture: Path, output: Path) -> dict:
    workspace, capture, output = workspace.resolve(), capture.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Case output already exists; use a new append-only artifact directory")
    production = RawArchive(workspace / "football_spider/platform_data")
    fixtures = {}
    for obs in production.observations("spftll"):
        if obs.http_status != 200:
            continue
        try:
            payload = json.loads(production.read_body(obs))
        except (ValueError, UnicodeError):
            continue
        for path, row in _walk(payload):
            source_id = str(row.get("matchId"))
            if source_id in {"3049653", "3049659"} and row.get("homeId") and row.get("awayId") and row.get("matchTime"):
                kickoff = stamp(row["matchTime"], "Asia/Shanghai")
                if obs.observed_at < kickoff:
                    fixtures.setdefault(source_id, (obs, path, row, kickoff))
    if "3049653" not in fixtures:
        raise ValueError("Existing Japan-Thailand pre-match fixture evidence was not found")
    archive = RawArchive(capture)
    source = [obs for obs in archive.observations("jfa") if obs.http_status == 200
              and obs.endpoint.endswith("/asiangames_2026_men/preview_03.html")]
    if not source:
        raise ValueError("Successful official preview archive is required")
    observation = source[0]
    body = archive.read_body(observation)
    fixture_obs, fixture_path, row, kickoff = fixtures["3049653"]
    league_key = hashlib.sha256(row["leagueChs"].encode("utf-8")).hexdigest()
    fixture = Fixture(match_id="spftll:match:3049653", competition_id="spftll:competition-name:" + league_key,
                      home_team_id="spftll:team:" + str(row["homeId"]), away_team_id="spftll:team:" + str(row["awayId"]),
                      home_team_name=row["homeChs"], away_team_name=row["awayChs"], kickoff=kickoff, season="2026")
    cutoff = kickoff - timedelta(hours=1)
    registry = CompetitionEvidenceRegistry(output / "registry", archive)
    specifications = [
        ("jfa:preview03:qualification:jpn:v1", "qualification", "qualified", "1試合を残した状況でノックアウトステージ進出を決め"),
        ("jfa:preview03:rank:jpn:v1", "rank_objective", "draw_sufficient_for_first_place", "引き分け以上で1位突破を確定させられる"),
    ]
    records = []
    for evidence_id, topic, value, anchor_text in specifications:
        needle = anchor_text.encode("utf-8")
        start = body.find(needle)
        if start < 0:
            raise ValueError("Official preview extraction anchor was not found in archived bytes")
        records.append(registry.register(
            evidence_id=evidence_id, fixture=fixture, observation=observation, document_scope="pre_match_preview",
            source_kind="official", claim={"topic": topic, "value": value, "team_id": fixture.home_team_id,
                                           "epistemic_status": "unverified"},
            anchor={"byte_start": start, "byte_end": start + len(needle), "sha256": hashlib.sha256(needle).hexdigest()},
            source_published_at=None, publication_time_basis="unknown"))
    ids = [record.evidence_id for record in records]
    operational = registry.project(fixture=fixture, prediction_cutoff=cutoff, acquired_evidence_ids=ids)
    hypothetical = {evidence_id: {"assumption_id": "round7-missing-publication-gate-probe",
                                  "assumed_available_at": cutoff.isoformat(),
                                  "basis": "source_publication_assumed_unchanged_since_publication"} for evidence_id in ids}
    historical = registry.project(fixture=fixture, prediction_cutoff=cutoff, acquired_evidence_ids=ids,
                                  information_mode="historical_reconstruction", reconstruction_assumptions=hypothetical)
    report = {"schema": "round7-real-cup-evidence-case-v1", "case": "Japan-Thailand Asian Games 2026-09-23",
              "fixture": fixture.model_dump(mode="json"), "prediction_cutoff": cutoff.isoformat(),
              "fixture_evidence": {"observation_id": fixture_obs.id, "observed_at": fixture_obs.observed_at.isoformat(),
                                   "content_sha256": fixture_obs.content_hash, "payload_path": fixture_obs.payload_path,
                                   "json_path": fixture_path, "source_match_id": "3049653"},
              "other_user_fixture_source_ids": sorted(set(fixtures) - {"3049653"}),
              "official_preview_observation": observation.model_dump(mode="json"),
              "source_published_at": None, "publication_missing_reason": "no_explicit_timestamp_in_captured_preview_html",
              "registrations": [record.model_dump(mode="json") for record in records],
              "operational_projection": operational.model_dump(mode="json"),
              "historical_gate_probe": historical.model_dump(mode="json"),
              "historical_gate_probe_basis": "hypothetical_timestamp_used_only_to_test_rejection_when_source_publication_unknown",
              "readiness": "REAL_RAW_CASE_RECORDED_NO_HISTORICAL_CONTEXT_TRAJECTORY",
              "real_operational_cup_context_policy_trajectories_added": 0,
              "claims_semantically_verified": 0, "policy_gain_ground_truth_added": 0}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "case.json").open("xb") as stream:
        stream.write(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.workspace, args.capture, args.output)
    print(json.dumps({"readiness": result["readiness"], "operational_exclusions": result["operational_projection"]["exclusions"],
                      "historical_exclusions": result["historical_gate_probe"]["exclusions"]}))


if __name__ == "__main__":
    main()
