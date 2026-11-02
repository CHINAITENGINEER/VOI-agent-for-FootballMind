"""Export normalized research projections into immutable ledger label inputs.

This module consumes ResearchDataset only. Provider parsing, raw archive hash
verification, entity mapping and revision resolution belong upstream. An export
never invents publication times or supersedes conflicting result evidence.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from football_research.baselines.api import canonical_json, content_hash, utc
from .types import ResearchDataset, ResearchMatch

SETTLEMENT_ADAPTER_VERSION = "research-settlement-v1"


def _observed_at(match: ResearchMatch, dataset: ResearchDataset):
    """Read normalized lineage timestamps, never provider payload fields."""
    if match.observed_at is not None:
        return match.observed_at, ['match.observed_at']
    times, bases = [], []
    per_match = dataset.audit.get("match_observed_at", {})
    if isinstance(per_match, dict) and match.match_id in per_match:
        times.append(utc(per_match[match.match_id]))
        bases.append("dataset.audit.match_observed_at")
    for reference in match.input_manifest:
        # spftll's research adapter emits this provider-neutral evidence string.
        # The raw source is not read, and fixture_available_at is not a status time.
        if "|at:" in reference:
            stamp = reference.rsplit("|at:", 1)[1].split("|", 1)[0]
            times.append(utc(stamp))
            bases.append("match.input_manifest.observation_at")
    if not times and dataset.audit.get("source_observed_at") is not None:
        times.append(utc(dataset.audit["source_observed_at"]))
        bases.append("dataset.audit.source_observed_at")
    return (max(times), sorted(set(bases))) if times else (None, [])


def export_results(dataset: ResearchDataset, match_ids: Iterable[str] | None = None):
    """Return (ledger_result_rows, coverage_report), with stable revision hashes.

    Finals require regulation scores and result_available_at. Cancelled,
    postponed and disputed states require an actual observation timestamp;
    suspended is exported as disputed. Scheduled/live/unknown remain pending.
    Every skip has an explicit reason. Information modes are preserved.
    """
    if not isinstance(dataset, ResearchDataset):
        raise TypeError("export_results requires a ResearchDataset")
    requested = None if match_ids is None else set(match_ids)
    if requested is not None and any(not isinstance(mid, str) or not mid for mid in requested):
        raise ValueError("match_ids must be nonempty strings")
    if isinstance(match_ids, str):
        raise ValueError("match_ids must be an iterable of IDs, not one string")
    now = datetime.now(timezone.utc)
    selected = [m for m in dataset.matches if requested is None or m.match_id in requested]
    rows, skipped, exported = [], [], []

    def skip(match, reason, detail=None):
        status_metadata = dataset.audit.get('match_settlement_metadata', {}).get(match.match_id, {})
        skipped.append({"match_id": match.match_id, "source_status": match.status,
                        "information_mode": match.information_mode, "reason": reason,
                        **({'platform_settlement': status_metadata} if status_metadata else {}),
                        **({"detail": detail} if detail else {})})

    for match in sorted(selected, key=lambda m: (m.kickoff, m.match_id)):
        if match.information_mode not in {"operational", "historical_reconstruction"}:
            skip(match, "unsupported_information_mode")
            continue
        if match.status not in {"finished", "cancelled", "postponed", "disputed", "suspended"}:
            skip(match, "no_settleable_status")
            continue
        if match.score_period != "regulation":
            skip(match, "regulation_semantics_missing")
            continue
        if not match.input_manifest or any(not isinstance(x, str) or not x for x in match.input_manifest):
            skip(match, "input_manifest_missing")
            continue
        if not match.source or match.source == "unknown":
            skip(match, "source_missing")
            continue
        try:
            observed, evidence_basis = _observed_at(match, dataset)
        except (ValueError, TypeError, AttributeError) as exc:
            skip(match, "invalid_observation_evidence", str(exc))
            continue
        if observed is None:
            skip(match, "observation_time_missing")
            continue
        if observed > now:
            skip(match, "future_observation")
            continue
        ingested, published = match.ingested_at, match.source_published_at
        operational_available = max(observed, ingested or observed)
        if match.operational_available_at is not None:
            if match.operational_available_at < operational_available:
                skip(match, 'operational_availability_before_evidence')
                continue
            operational_available = match.operational_available_at
        if ingested is not None and ingested > now:
            skip(match, 'future_ingestion')
            continue
        if published is not None and published > observed:
            skip(match, 'publication_after_observation')
            continue
        explicit_clocks = any(value is not None for value in
                              (match.observed_at, ingested, published, match.operational_available_at))
        status = "final" if match.status == "finished" else "disputed" if match.status == "suspended" else match.status
        if status == "final":
            if match.outcome is None or match.result_available_at is None:
                skip(match, "final_label_or_availability_missing")
                continue
            available = match.result_available_at
            if not explicit_clocks and available > observed:
                skip(match, "availability_after_observation")
                continue
            if explicit_clocks and match.information_mode == 'operational' and available < operational_available:
                skip(match, 'operational_availability_before_evidence')
                continue
        else:
            # Earliest fixture visibility is not evidence of when its status changed.
            available = operational_available
        if available > now or operational_available > now:
            skip(match, 'future_availability')
            continue
        assumptions = dataset.audit.get("assumptions")
        if match.information_mode == "historical_reconstruction":
            if not assumptions and "ASSUMED_" not in match.time_basis and "assumed_" not in match.time_basis:
                skip(match, "reconstruction_assumption_missing")
                continue
        payload = {"match_id": match.match_id, "information_mode": match.information_mode,
                   "status": status, "score_period": "regulation", "kickoff": match.kickoff.isoformat(),
                   "home_goals": match.home_goals if status == "final" else None,
                   "away_goals": match.away_goals if status == "final" else None,
                   "source": match.source,
                   "source_reference": f"research:{dataset.dataset_version}|manifest_sha256:{content_hash(match.input_manifest)}",
                   "dataset_version": dataset.dataset_version, "input_manifest": list(match.input_manifest),
                   "available_at": available.isoformat(), "observed_at": observed.isoformat(),
                   "supersedes": [], "adapter_version": SETTLEMENT_ADAPTER_VERSION,
                   "source_status": match.status, "observation_time_basis": evidence_basis}
        if explicit_clocks:
            payload.update(time_contract_version='separate-clocks-v2',
                           operational_available_at=operational_available.isoformat(),
                           ingested_at=ingested.isoformat() if ingested else None,
                           source_published_at=published.isoformat() if published else None)
        if match.information_mode == "historical_reconstruction":
            payload["availability_assumption"] = canonical_json({"time_basis": match.time_basis,
                        "assumptions": assumptions or {},
                        "source_revision_bias": dataset.audit.get("source_revision_bias", "Current observed data may include later revisions"),
                        "nonfinal_status_timing": "actual_observation" if status != "final" else None})
        if match.status == "suspended":
            payload["reason"] = "Normalized suspended fixture requires resolution before final scoring"
        payload["revision_id"] = f"{SETTLEMENT_ADAPTER_VERSION}:{content_hash(payload)}"
        rows.append(payload)
        exported.append({"match_id": match.match_id, "status": status,
                         "revision_id": payload["revision_id"], "information_mode": match.information_mode,
                         "available_at": payload["available_at"], "observed_at": payload["observed_at"],
                         "observation_time_basis": evidence_basis})
    found = {m.match_id for m in selected}
    report = {"adapter_version": SETTLEMENT_ADAPTER_VERSION, "dataset_version": dataset.dataset_version,
              "dataset_matches": len(dataset.matches), "requested_match_count": len(requested) if requested is not None else None,
              "selected_matches": len(selected), "exported_results": len(rows),
              "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
              "information_mode_counts": dict(sorted(Counter(r["information_mode"] for r in rows).items())),
              "skipped_matches": len(skipped), "skip_reason_counts": dict(sorted(Counter(r["reason"] for r in skipped).items())),
              "missing_match_ids": sorted(requested - found) if requested is not None else [],
              "rows_sha256": content_hash(rows), "exported": exported, "skipped": skipped,
              "revision_policy": "No inferred supersession; changed contradictory projections remain conflicts until explicitly resolved",
              "evidence_policy": "Upstream normalized evidence only; no provider parsing or inferred historical publication"}
    return rows, report
