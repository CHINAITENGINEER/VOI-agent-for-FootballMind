"""Issue new forecasts using the real clock, without changing old artifacts."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .api import PredictionService, canonical_json, content_hash, training_rows
from football_research.data_adapters import load_spftll_archive, save_jsonl
from football_research.evaluation.ledger import ForecastLedger
from football_research.evaluation.runner import quote_features


def issue_batch(*, archive_root, ledger_path, output_dir, models=("frequency", "poisson", "dixon_coles", "market"), limit=20):
    if limit <= 0 or any(name not in {"frequency", "poisson", "dixon_coles", "market"} for name in models):
        raise ValueError("Positive limit and fixture/market model names required; feature issuance uses a timestamped snapshot")
    cutoff = datetime.now(timezone.utc)
    output = Path(output_dir) / (cutoff.strftime("%Y%m%dT%H%M%S") + "-" + str(uuid4()))
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_spftll_archive(archive_root, as_of=cutoff)
    save_jsonl(dataset, output / "dataset.jsonl")
    matches, odds = [asdict(m) for m in dataset.matches], [asdict(q) for q in dataset.odds]
    eligible = training_rows(matches, as_of=cutoff, information_mode="operational")
    fixtures = sorted([m for m in matches if m["status"] == "scheduled" and m["score_period"] == "regulation"
                       and m["fixture_available_at"] is not None and m["fixture_available_at"] <= cutoff < m["kickoff"]],
                      key=lambda m: (m["kickoff"], m["match_id"]))[:limit]
    ledger, issued, rejected, runs = ForecastLedger(ledger_path), [], [], []
    for competition in sorted({m["competition_id"] for m in fixtures}):
        history = [m for m in eligible if m["competition_id"] == competition]
        for name in models:
            try:
                service = PredictionService.train(name, history, as_of=cutoff, dataset_version=dataset.dataset_version)
                path = output / "models" / f"{content_hash(competition)[:10]}-{name}.json"
                service.save(path)
                runs.append({"model": name, "competition_id": competition, "artifact": str(path), "metadata": service.describe()})
            except (ValueError, RuntimeError) as exc:
                rejected.append({"competition_id": competition, "model": name, "reason": str(exc)})
                continue
            for fixture in [m for m in fixtures if m["competition_id"] == competition]:
                features = quote_features(fixture, cutoff, odds) if name == "market" else None
                try:
                    operation = ledger.issue_forecast(service, fixture, prediction_cutoff=cutoff, features=features)
                    issued.append({"forecast_id": operation["forecast_id"], "match_id": fixture["match_id"],
                                   "model": name, "issued_at": operation["record"]["issued_at"],
                                   "kickoff": fixture["kickoff"].isoformat(), "status": operation["status"]})
                except (ValueError, RuntimeError, KeyError) as exc:
                    rejected.append({"match_id": fixture["match_id"], "model": name, "reason": str(exc)})
    score = ledger.score()
    (output / "pending.json").write_text(canonical_json(score) + "\n", encoding="utf-8")
    report = {"status": "REGISTERED" if issued else "NO_ELIGIBLE_FORECASTS", "registration_kind": "prospective",
              "information_mode": "operational", "prediction_cutoff": cutoff.isoformat(),
              "generated_and_registered_at": "per-forecast system clock issued_at in immutable ledger",
              "dataset_version": dataset.dataset_version, "dataset_audit": dataset.audit,
              "eligible_matches": len(fixtures), "issued_predictions": len(issued), "issued_matches": len({r["match_id"] for r in issued}),
              "forecasts": issued, "rejected": rejected, "model_runs": runs, "ledger_path": str(Path(ledger_path).resolve()),
              "output_dir": str(output.resolve()), "settlement_status": score["status_counts"],
              "source_freshness_note": "Archived fixture and quote evidence; no new external request by research task."}
    (output / "issuance-report.json").write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
