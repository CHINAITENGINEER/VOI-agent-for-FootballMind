"""Run with python -m football_research.baselines from the workspace root."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import importlib.metadata
import json
from pathlib import Path
import sys

from .api import PredictionService, canonical_json, content_hash, training_rows, utc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def load(path):
    from football_research.data_adapters import load_jsonl
    dataset = load_jsonl(path)
    return dataset, [asdict(r) for r in dataset.matches], [asdict(r) for r in dataset.odds]


def feature_options(parser, *, train=False):
    parser.add_argument("--feature-file", help="Batch feature JSONL or complete exported feature directory")
    parser.add_argument("--feature-version", help="Exact version required for a feature join")
    if train:
        parser.add_argument("--lead-hours", type=float, default=24)
    parser.add_argument("--model-parameters", help="JSON file; model kwargs for train, name-to-kwargs for backtest")


def read_feature_index(args, service=None):
    if not args.feature_file:
        return None
    if not args.feature_version:
        raise ValueError("--feature-version is required with --feature-file")
    from football_research.data_adapters.batch_features import load_feature_index
    parameters = read_parameters(args)
    fields = set(parameters.get("feature_names", []))
    for value in parameters.values():
        if isinstance(value, dict):
            fields.update(value.get("feature_names", []))
    if service is not None and service.model_name in {"logistic", "tree"}:
        fields.update(service.model.feature_names)
    return load_feature_index(args.feature_file, allowed_features=fields)


def read_parameters(args):
    return json.loads(Path(args.model_parameters).read_text(encoding="utf-8")) if args.model_parameters else {}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"register-forecasts", "settle-forecasts", "score-forecasts"}:
        from football_research.evaluation.ledger_cli import main as ledger_main
        return ledger_main(arguments)
    parser = argparse.ArgumentParser(description="FootballMind regulation 1X2 baselines")
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("list-models", help="Agent-consumable model input requirements")
    metadata.add_argument("--artifact")
    history = commands.add_parser("import-history", help="Consume a completed platform normalized history export")
    history.add_argument("--platform-export", required=True)
    history.add_argument("--output", required=True)
    result_export = commands.add_parser("export-results", help="Project evidenced dataset outcomes into append-only settlement rows")
    result_export.add_argument("--dataset", required=True)
    result_export.add_argument("--ledger", help="Restrict export to frozen match IDs in this ledger")
    result_export.add_argument("--output", required=True)
    demo = commands.add_parser("run-local", help="Read actual archives, train and predict; report strict backtest coverage")
    demo.add_argument("--archive-root", default="football_spider/platform_data/raw")
    demo.add_argument("--dataset", help="Replay a previously frozen dataset JSONL")
    demo.add_argument("--output", default="var/parallel/baselines/real")
    demo.add_argument("--cutoff", default="2026-09-20T11:22:02Z", help="Explicit frozen UTC experiment cutoff")
    demo.add_argument("--models", nargs="+", default=["frequency", "poisson", "dixon_coles", "market"])
    train = commands.add_parser("train")
    train.add_argument("--dataset", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--as-of", required=True)
    train.add_argument("--competition")
    train.add_argument("--output", required=True)
    feature_options(train, train=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--dataset", required=True)
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--match-id", required=True)
    predict.add_argument("--cutoff", required=True)
    predict.add_argument("--features", help="Optional explicit feature snapshot JSON (Elo/market)")
    predict.add_argument("--output", required=True)
    feature_options(predict)
    batch = commands.add_parser("batch-predict")
    batch.add_argument("--dataset", required=True)
    batch.add_argument("--artifact", required=True)
    batch.add_argument("--lead-hours", nargs="+", type=float, default=[24])
    batch.add_argument("--output", required=True)
    feature_options(batch)
    backtest = commands.add_parser("backtest")
    backtest.add_argument("--dataset", required=True)
    backtest.add_argument("--validation-start", required=True)
    backtest.add_argument("--validation-days", type=int, default=30)
    backtest.add_argument("--test-days", type=int, default=30)
    backtest.add_argument("--lead-hours", nargs="+", type=float, default=[24, 1])
    backtest.add_argument("--models", nargs="+", default=["frequency", "poisson", "dixon_coles", "market"])
    backtest.add_argument("--min-train-matches", type=int, default=20)
    backtest.add_argument("--output", required=True)
    feature_options(backtest)
    issue = commands.add_parser("issue-forecasts", help="Generate now and register prospective predictions")
    issue.add_argument("--archive-root", default="football_spider/platform_data/raw")
    issue.add_argument("--ledger", required=True)
    issue.add_argument("--output", required=True)
    issue.add_argument("--models", nargs="+", default=["frequency", "poisson", "dixon_coles", "market"])
    issue.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(arguments)
    if args.command == "list-models":
        from .metadata import model_metadata
        report = PredictionService.load(args.artifact).describe() if args.artifact else model_metadata()
    elif args.command == "import-history":
        from football_research.data_adapters.platform_history import load_completed_history_export
        from football_research.data_adapters import save_jsonl
        dataset = load_completed_history_export(args.platform_export)
        save_jsonl(dataset, args.output)
        report = {"status": "IMPORTED", "matches": len(dataset.matches), "dataset_version": dataset.dataset_version,
                  "audit": dataset.audit, "output": args.output}
    elif args.command == "export-results":
        from football_research.data_adapters.settlement import export_results
        from football_research.evaluation.ledger import ForecastLedger
        dataset, _, _ = load(args.dataset)
        ids = {r["prediction"]["match_id"] for r in ForecastLedger(args.ledger).forecasts()} if args.ledger else None
        rows, report = export_results(dataset, match_ids=ids)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
        write_json(target.with_suffix(".report.json"), report)
    elif args.command == "issue-forecasts":
        from .prospective import issue_batch
        report = issue_batch(archive_root=args.archive_root, ledger_path=args.ledger, output_dir=args.output,
                             models=args.models, limit=args.limit)
    elif args.command == "run-local":
        report = run_local(args)
    elif args.command == "train":
        dataset, matches, _ = load(args.dataset)
        matches = [r for r in matches if args.competition is None or r["competition_id"] == args.competition]
        mode = unique_mode(matches)
        parameters = read_parameters(args)
        assembly = None
        if args.model in {"elo", "logistic", "tree"}:
            from football_research.evaluation.runner import model_feature_names
            index = read_feature_index(args)
            if index is None:
                raise ValueError("Feature model training requires --feature-file and --feature-version")
            matches, assembly = index.assemble_training_rows(matches, feature_version=args.feature_version,
                lead_hours=args.lead_hours, information_mode=mode,
                required_features=model_feature_names(args.model, parameters), allow_missing_values=args.model in {"logistic", "tree"})
        service = PredictionService.train(args.model, matches, as_of=args.as_of,
                                          dataset_version=dataset.dataset_version, information_mode=mode, parameters=parameters)
        service.save(args.output)
        report = {"status": "OK", "artifact": args.output, "training_matches": service.training_manifest["training_matches"],
                  "feature_assembly": assembly, "model_metadata": service.describe()}
    elif args.command == "predict":
        from football_research.evaluation.runner import quote_features
        _, matches, odds = load(args.dataset)
        fixture = next(r for r in matches if r["match_id"] == args.match_id)
        service = PredictionService.load(args.artifact)
        features = json.loads(Path(args.features).read_text(encoding="utf-8")) if args.features else None
        if args.feature_file:
            if features is not None:
                raise ValueError("Use either --features or --feature-file")
            index = read_feature_index(args, service)
            features = index.join(fixture, prediction_cutoff=args.cutoff, information_mode=service.information_mode,
                feature_version=args.feature_version, required_features=service.describe()["required_features"],
                allow_missing_values=service.model_name in {"logistic", "tree"})
        if features is None and service.model_name == "market":
            features = quote_features(fixture, args.cutoff, odds)
        prediction = service.predict(fixture, prediction_cutoff=args.cutoff, features=features)
        report = prediction.model_dump(mode="json")
        write_json(args.output, report)
    elif args.command == "batch-predict":
        report = batch_predict(args)
    else:
        from football_research.evaluation.runner import evaluate
        dataset, matches, odds = load(args.dataset)
        report = evaluate(matches, dataset_version=dataset.dataset_version, information_mode=unique_mode(matches),
            output_dir=args.output, initial_validation_start=args.validation_start, validation_days=args.validation_days,
            test_days=args.test_days, models=args.models, lead_hours=args.lead_hours, odds=odds,
            min_train_matches=args.min_train_matches, evidence_grade=dataset.audit.get("evidence_grade", "functional_validation"),
            feature_index=read_feature_index(args), feature_version=args.feature_version,
            model_parameters=read_parameters(args), dataset_audit=dataset.audit)
    print(canonical_json(report))


def batch_predict(args):
    from football_research.evaluation.runner import quote_features
    _, matches, odds = load(args.dataset)
    model = PredictionService.load(args.artifact)
    index = read_feature_index(args, model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results, rejected = [], []
    for fixture in matches:
        for lead in args.lead_hours:
            cutoff = utc(fixture["kickoff"]) - timedelta(hours=lead)
            try:
                features = None
                if model.model_name in {"elo", "logistic", "tree"}:
                    if index is None:
                        raise ValueError("Feature model requires an explicit feature index")
                    features = index.join(fixture, prediction_cutoff=cutoff, information_mode=model.information_mode,
                        feature_version=args.feature_version, required_features=model.describe()["required_features"],
                        allow_missing_values=model.model_name in {"logistic", "tree"})
                elif model.model_name == "market":
                    features = quote_features(fixture, cutoff, odds)
                prediction = model.predict(fixture, prediction_cutoff=cutoff, features=features)
                results.append({**prediction.model_dump(mode="json"), "kickoff": utc(fixture["kickoff"]).isoformat(), "lead_hours": lead})
            except (ValueError, KeyError) as exc:
                rejected.append({"match_id": fixture["match_id"], "lead_hours": lead, "reason": str(exc)})
    (output / "predictions.jsonl").write_text("".join(canonical_json(r) + "\n" for r in results), encoding="utf-8")
    report = {"status": "OK" if results else "NO_ELIGIBLE_PREDICTIONS", "predictions": len(results),
              "rejected": rejected, "model_metadata": model.describe(), "feature_version": args.feature_version}
    write_json(output / "report.json", report)
    return report


def unique_mode(matches):
    modes = {r["information_mode"] for r in matches}
    if len(modes) != 1:
        raise ValueError("Exactly one information mode required")
    return modes.pop()


def run_local(args):
    from football_research.data_adapters import load_spftll_archive, save_jsonl
    from football_research.evaluation.runner import evaluate, quote_features
    if args.dataset:
        from football_research.data_adapters import load_jsonl
        dataset = load_jsonl(args.dataset)
    else:
        dataset = load_spftll_archive(Path(args.archive_root), information_mode="operational", as_of=args.cutoff)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(dataset, output / "dataset.jsonl")
    matches, odds = [asdict(r) for r in dataset.matches], [asdict(r) for r in dataset.odds]
    cutoff = utc(args.cutoff)
    eligible = training_rows(matches, as_of=cutoff, information_mode="operational")
    fixtures = [r for r in matches if r["score_period"] == "regulation" and r["status"] == "scheduled"
                and r["fixture_available_at"] is not None and r["fixture_available_at"] <= cutoff < r["kickoff"]]
    forecasts, runs, skips = [], [], []
    for competition in sorted({r["competition_id"] for r in fixtures}):
        history = [r for r in eligible if r["competition_id"] == competition]
        for name in args.models:
            try:
                service = PredictionService.train(name, history, as_of=cutoff, dataset_version=dataset.dataset_version,
                                                   information_mode="operational")
                artifact = output / "models" / f"{content_hash(competition)[:10]}-{name}.json"
                service.save(artifact)
                restored = PredictionService.load(artifact)
            except (ValueError, RuntimeError) as exc:
                skips.append({"model": name, "competition_id": competition, "reason": str(exc)})
                continue
            run = {"model": name, "competition_id": competition, "training_matches": service.training_manifest["training_matches"],
                   "artifact": str(artifact), "predictions": 0, "json_reload_equivalent": True}
            for fixture in [r for r in fixtures if r["competition_id"] == competition]:
                features = quote_features(fixture, cutoff, odds) if name == "market" else None
                try:
                    prediction = service.predict(fixture, prediction_cutoff=cutoff, features=features)
                    replay = restored.predict(fixture, prediction_cutoff=cutoff, features=features)
                    if prediction != replay:
                        raise RuntimeError("Reloaded model prediction differs")
                    forecasts.append({**prediction.model_dump(mode="json"), "model": name,
                                      "home_team_name": fixture.get("home_team_name"), "away_team_name": fixture.get("away_team_name"),
                                      "competition_name": fixture.get("competition_name"), "kickoff": fixture["kickoff"].isoformat(),
                                      "lead_hours": (fixture["kickoff"] - cutoff).total_seconds() / 3600})
                    run["predictions"] += 1
                except (ValueError, RuntimeError, KeyError) as exc:
                    skips.append({"model": name, "match_id": fixture["match_id"], "reason": str(exc)})
            runs.append(run)
    (output / "predictions.jsonl").write_text("".join(canonical_json(r) + "\n" for r in forecasts), encoding="utf-8")
    earliest = min((r["kickoff"] for r in eligible), default=cutoff)
    backtest = evaluate(matches, dataset_version=dataset.dataset_version, information_mode="operational",
        output_dir=output / "strict-backtest", initial_validation_start=earliest + timedelta(days=60),
        models=args.models, odds=odds, evidence_grade="real_observation_time_audit")
    report = {"status": "FUNCTIONAL_VALIDATION" if forecasts else "NO_ELIGIBLE_PREDICTIONS",
              "dataset_version": dataset.dataset_version, "cutoff": cutoff.isoformat(),
              "information_mode": "operational", "training_outcomes": len(eligible),
              "eligible_future_matches": len(fixtures), "prediction_rows": len(forecasts),
              "model_runs": runs, "skips": skips, "data_audit": dataset.audit,
              "strict_backtest": {k: backtest[k] for k in ("status", "candidate_snapshots", "test_matches", "metrics", "exclusions")},
              "environment": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pydantic", "matplotlib")},
              "limitation": "Real records validate training and forecast execution; no retrospective accuracy claimed without historical visibility."}
    write_json(output / "run-report.json", report)
    return report


if __name__ == "__main__":
    main()
