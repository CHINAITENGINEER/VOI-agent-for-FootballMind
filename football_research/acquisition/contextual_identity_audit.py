"""Audit cross-export identity continuity using provider IDs, without outcomes."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def audit(earlier_bundle, current_bundle):
    inputs, indexes, active = {}, {}, {}
    for role, directory in (("earlier", earlier_bundle), ("current", current_bundle)):
        path = Path(directory)/"id_mappings.jsonl"
        mapping = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["entity_type"] not in {"team", "competition"}:
                continue
            key = (row["source"], row["entity_type"], str(row["source_id"]))
            if key in mapping:
                raise ValueError("Ambiguous repeated source identity: "+repr(key))
            mapping[key] = {k: row.get(k) for k in ("entity_id", "status", "valid_from", "valid_to", "version", "evidence")}
        indexes[role] = mapping
        match_path = Path(directory)/"matches.jsonl"
        # Immediately project identity fields; no score, feature, result,
        # chronology or model-performance field participates in this audit.
        fixture_ids = [{k: row[k] for k in ("home_team_id", "away_team_id", "competition_id")}
                       for row in (json.loads(line) for line in match_path.read_text(encoding="utf-8").splitlines())]
        active_entities = {v for row in fixture_ids for v in row.values()}
        active[role] = {key for key, item in mapping.items() if item["entity_id"] in active_entities}
        inputs[role] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
                       "team_keys": sum(k[1] == "team" for k in mapping),
                       "competition_keys": sum(k[1] == "competition" for k in mapping),
                       "fixture_identity_source": str(match_path.resolve()), "fixture_identity_sha256": sha256(match_path.read_bytes()).hexdigest(),
                       "active_team_keys": sum(k[1] == "team" for k in active[role])}
    shared = indexes["earlier"].keys() & indexes["current"].keys()
    records = []
    for key in sorted(shared):
        a, b = indexes["earlier"][key], indexes["current"][key]
        records.append({"provider": key[0], "entity_type": key[1], "source_id": key[2],
                        "same_export_entity_id": a["entity_id"] == b["entity_id"], "active_in_both_seasons": key in active["earlier"] and key in active["current"],
                        "earlier": a, "current": b})
    return {"status": "VERIFIED_SOURCE_ID_CONTINUITY_AUDIT", "inputs": inputs,
        "counts": {kind: {"shared_provider_keys": sum(r["entity_type"] == kind for r in records),
                          "same_canonical_id": sum(r["entity_type"] == kind and r["same_export_entity_id"] for r in records),
                          "different_canonical_id": sum(r["entity_type"] == kind and not r["same_export_entity_id"] for r in records)}
                   for kind in ("team", "competition")},
        "shared_id_records": records,
        "active_counts": {kind: {"shared_provider_keys": sum(r["entity_type"] == kind and r["active_in_both_seasons"] for r in records),
                                 "different_canonical_id": sum(r["entity_type"] == kind and r["active_in_both_seasons"] and not r["same_export_entity_id"] for r in records)}
                          for kind in ("team", "competition")},
        "earlier_only": [list(k) for k in sorted(indexes["earlier"].keys()-indexes["current"].keys())],
        "current_only": [list(k) for k in sorted(indexes["current"].keys()-indexes["earlier"].keys())],
        "interpretation": "same provider/entity_type/source_id maps to different export canonical UUID; exact metadata join, no fuzzy-name inference",
        "files_read": "id_mappings.jsonl and only home_team_id/away_team_id/competition_id projection of matches.jsonl; no outcome/feature field used",
        "repair_scope_proposal": "frozen explicit provider-ID crosswalk at value feature boundary; keep original predictor/export/OOF labels unchanged",
        "repair_applied": False,
        "caveat": "mapping snapshots were observed in 2026; historical reconstruction uses structural identity equivalence, not historical publication-time proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier-bundle", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.earlier_bundle, args.bundle)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
