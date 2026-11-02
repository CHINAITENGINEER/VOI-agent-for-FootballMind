"""Frozen rolling-window evaluation with auditable absence of eligible data."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from itertools import combinations, product
from pathlib import Path
from typing import Mapping, Sequence

from football_research.baselines.api import PredictionService, canonical_json, content_hash, utc
from .metrics import score_predictions, paired_model_comparison, reliability_table, save_calibration_plot
from .splits import rolling_origin_splits

FEATURE_MODELS = frozenset({"elo", "logistic", "tree"})


def model_feature_names(name, parameters=None):
    if name == "elo":
        return ("home_elo", "away_elo")
    if name in {"logistic", "tree"}:
        return tuple((parameters or {}).get("feature_names", ("home_elo", "away_elo")))
    return ()


def make_snapshots(matches: Sequence[Mapping], lead_hours=(24., 1.)):
    snapshots, reasons = [], Counter()
    for match in matches:
        if match.get("score_period") != "regulation":
            reasons["unknown_or_non_regulation_period"] += 1
            continue
        if match.get("status") in {"cancelled", "postponed", "suspended"}:
            reasons["inactive_fixture"] += 1
            continue
        if match.get("home_goals") is None or match.get("away_goals") is None:
            reasons["no_final_label"] += 1
            continue
        if match.get("result_available_at") is None:
            reasons["unknown_label_availability"] += 1
            continue
        for lead in lead_hours:
            if lead <= 0:
                raise ValueError("Prematch lead hours must be positive")
            cutoff = utc(match["kickoff"]) - timedelta(hours=lead)
            if match.get("fixture_available_at") is None or utc(match["fixture_available_at"]) > cutoff:
                reasons["fixture_not_known_at_requested_cutoff"] += 1
                continue
            snapshots.append(dict(match, kickoff=utc(match["kickoff"]), prediction_cutoff=cutoff, lead_hours=float(lead)))
    return snapshots, dict(reasons)


def quote_features(fixture, cutoff, odds):
    """Pick one complete immutable market; never mix selections across snapshots."""
    cutoff = utc(cutoff)
    candidates = [q for q in odds if str(q["match_id"]) == str(fixture["match_id"])
                  and q.get("period") == "regulation"
                  and q.get("information_mode", "operational") == fixture["information_mode"]
                  and utc(q["observed_at"]) <= cutoff
                  and (q.get("source_timestamp") is None or utc(q["source_timestamp"]) <= cutoff)]
    if not candidates:
        return None
    # Provider and bookmaker are a deterministic tie-break, not retrospective best odds.
    quote = max(candidates, key=lambda q: (utc(q["observed_at"]), q.get("provider", ""), str(q.get("bookmaker")), content_hash(q)))
    return {"match_id": fixture["match_id"], "prediction_cutoff": cutoff,
            "information_mode": fixture["information_mode"], "input_manifest": {
                "quote_refs": list(quote["input_manifest"]), "provider": quote.get("provider"),
                "bookmaker": quote.get("bookmaker"), "observed_at": utc(quote["observed_at"]).isoformat()},
            "odds_home": quote["home_price"], "odds_draw": quote["draw_price"], "odds_away": quote["away_price"],
            "odds_available_at": utc(quote["observed_at"]), "odds_source_timestamp": quote.get("source_timestamp"),
            "odds_period": "regulation", "odds_market_type": "1x2", "odds_snapshot_id": content_hash(quote)}


def evaluate(matches, *, dataset_version, information_mode, output_dir,
             initial_validation_start, validation_days=30, test_days=30, step_days=None,
             models=("frequency", "poisson", "dixon_coles", "market"), lead_hours=(24., 1.),
             odds=(), min_train_matches=20, evidence_grade="functional_validation",
             feature_index=None, feature_version=None, model_parameters=None, dataset_audit=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not lead_hours or len(set(lead_hours)) != len(lead_hours):
        raise ValueError("Supply distinct positive prediction lead hours")
    model_parameters = model_parameters or {}
    snapshots, exclusions = make_snapshots(matches, lead_hours)
    # Training can use completed earlier matches even if their own prematch
    # fixture had not yet been observed: no retroactive prediction is made for them.
    records = list(snapshots)
    snapshot_ids = {r["match_id"] for r in snapshots}
    records.extend(dict(r, prediction_cutoff=utc(r["kickoff"]) - timedelta(hours=max(lead_hours)))
                   for r in matches if r["match_id"] not in snapshot_ids and r.get("score_period") == "regulation"
                   and r.get("result_available_at") is not None and r.get("home_goals") is not None)
    folds = rolling_origin_splits(records, initial_validation_start=utc(initial_validation_start),
                                 validation_window=timedelta(days=validation_days),
                                 test_window=timedelta(days=test_days),
                                 step=timedelta(days=step_days or test_days)) if records else []
    forecasts, runs, skips = [], [], []
    consumed = set()
    for index, fold in enumerate(folds):
        # Only candidate snapshots with proven fixture availability can be scored.
        test = [r for r in fold.test if (r["match_id"] in snapshot_ids and "lead_hours" in r)]
        validation = [r for r in fold.validation if (r["match_id"] in snapshot_ids and "lead_hours" in r)]
        if not test:
            continue
        for competition in sorted({r["competition_id"] for r in test}):
            train_base = [r for r in fold.train if r["competition_id"] == competition]
            for name, lead in product(models, lead_hours):
                test_comp = [r for r in test if r["competition_id"] == competition and r["lead_hours"] == lead]
                valid_comp = [r for r in validation if r["competition_id"] == competition and r["lead_hours"] == lead]
                if not test_comp:
                    continue
                train = list({r["match_id"]: r for r in train_base}.values())
                requirements = model_feature_names(name, model_parameters.get(name))
                run = {"fold": index, "competition_id": competition, "model": name,
                       "lead_hours": float(lead), "feature_version": feature_version if name in FEATURE_MODELS else None,
                       "train_cutoff": fold.train_cutoff.isoformat(),
                       "training_matches": len({r["match_id"] for r in train}),
                       "validation_matches": len({r["match_id"] for r in valid_comp}),
                       "test_matches": len({r["match_id"] for r in test_comp}),
                       "tuning": "fixed_parameters_no_tuning_or_calibration", "split": fold.manifest()}
                if name in FEATURE_MODELS:
                    if feature_index is None or feature_version is None:
                        skips.append(dict(run, reason="feature_index_and_explicit_version_required"))
                        continue
                    train, assembly = feature_index.assemble_training_rows(train, feature_version=feature_version,
                        lead_hours=lead, information_mode=information_mode, required_features=requirements,
                        allow_missing_values=name in {"logistic", "tree"})
                    run["feature_training_assembly"] = assembly
                    run["training_matches"] = len({r["match_id"] for r in train})
                if run["training_matches"] < min_train_matches and name != "market":
                    skips.append(dict(run, reason="insufficient_training_matches"))
                    continue
                try:
                    service = PredictionService.train(name, train, as_of=fold.train_cutoff,
                        dataset_version=dataset_version, information_mode=information_mode,
                        parameters=model_parameters.get(name),
                        excluded_match_ids={r["match_id"] for r in valid_comp + test_comp})
                    run["training_matches"] = service.training_manifest["training_matches"]
                    artifact_path = output / "models" / f"fold-{index}-{content_hash(competition)[:10]}-{name}-{lead:g}h.json"
                    service.save(artifact_path)
                    run["model_artifact"] = str(artifact_path)
                except (ValueError, RuntimeError) as exc:
                    skips.append(dict(run, reason=str(exc)))
                    continue
                for partition, targets in (("validation", valid_comp), ("test", test_comp)):
                    for row in targets:
                        key = (name, row["match_id"], row["prediction_cutoff"], partition)
                        # Overlapping rolling test windows must not inflate sample size.
                        if key in consumed:
                            continue
                        features = quote_features(row, row["prediction_cutoff"], odds) if name == "market" else None
                        try:
                            if name in FEATURE_MODELS:
                                features = feature_index.join(row, prediction_cutoff=row["prediction_cutoff"],
                                    information_mode=information_mode, feature_version=feature_version,
                                    required_features=requirements, allow_missing_values=name in {"logistic", "tree"})
                            prediction = service.predict(row, prediction_cutoff=row["prediction_cutoff"], features=features)
                        except (ValueError, RuntimeError, KeyError) as exc:
                            skips.append({"fold": index, "model": name, "match_id": row["match_id"],
                                          "partition": partition, "lead_hours": lead, "reason": str(exc)})
                            continue
                        consumed.add(key)
                        forecasts.append({**prediction.model_dump(mode="json"), "model": name, "fold": index,
                                          "partition": partition, "kickoff": utc(row["kickoff"]).isoformat(),
                                          "lead_hours": row["lead_hours"], "home_goals": row["home_goals"],
                                          "away_goals": row["away_goals"], "competition_id": competition})
                runs.append(run)
    scores = {}
    grouped = defaultdict(list)
    for row in forecasts:
        grouped[(row["partition"], row["model"], row["lead_hours"])].append(row)
    for (partition, name, lead), rows in sorted(grouped.items()):
        key = f"{partition}/{name}/{lead:g}h"
        scores[key] = score_predictions(rows)
        if partition == "test":
            save_calibration_plot(rows, output / f"calibration-{name}-{lead:g}h.png")
            (output / f"calibration-{name}-{lead:g}h.json").write_text(canonical_json(reliability_table(rows)) + "\n", encoding="utf-8")
    comparisons = {}
    common_scores, coverage = {}, {}
    for lead in lead_hours:
        indices = {name: {(r["match_id"], r["prediction_cutoff"]): r for r in grouped[("test", name, float(lead))]}
                   for name in models}
        active = {name: rows for name, rows in indices.items() if rows}
        shared = set.intersection(*(set(rows) for rows in active.values())) if active else set()
        common_scores[f"{lead:g}h"] = {"participating_models": list(active),
            "unavailable_models": [name for name in models if not indices[name]],
            "common_matches": len({key[0] for key in shared}), "common_predictions_per_model": len(shared),
            "all_requested_models_common_predictions": len(shared) if len(active) == len(models) else 0,
            "metrics": {name: score_predictions([rows[k] for k in sorted(shared)]) for name, rows in active.items()}}
        for name in models:
            coverage[f"{name}/{lead:g}h"] = {"test_predictions": len(indices[name]),
                "test_matches": len({k[0] for k in indices[name]}),
                "no_predictions_reason": "see fold/skips or missing market/features" if not indices[name] else None}
        for left, right in combinations(models, 2):
            lhs, rhs = grouped[("test", left, float(lead))], grouped[("test", right, float(lead))]
            if lhs and rhs:
                pair_keys = indices[left].keys() & indices[right].keys()
                comparisons[f"{left}_minus_{right}/{lead:g}h"] = {**paired_model_comparison(lhs, rhs),
                    "left_common_metrics": score_predictions([indices[left][k] for k in sorted(pair_keys)]),
                    "right_common_metrics": score_predictions([indices[right][k] for k in sorted(pair_keys)])}
    report = {"status": "OK" if any(r["partition"] == "test" for r in forecasts) else "NO_ELIGIBLE_TEST_PREDICTIONS",
              "evidence_grade": evidence_grade, "dataset_version": dataset_version, "information_mode": information_mode,
              "dataset_audit": dataset_audit or {}, "feature_version": feature_version,
              "target": "regulation_plus_stoppage_excludes_extra_time_penalties",
              "input_matches": len({r["match_id"] for r in matches}), "candidate_snapshots": len(snapshots),
              "exclusions": exclusions, "configuration": {"initial_validation_start": utc(initial_validation_start).isoformat(),
                  "validation_days": validation_days, "test_days": test_days, "step_days": step_days or test_days,
                  "lead_hours": list(lead_hours), "min_train_matches": min_train_matches, "models": list(models),
                  "model_parameters": model_parameters, "feature_training_lead_policy": "fit_separately_at_each_prediction_lead"},
              "fold_runs": runs, "skips": skips, "metrics": scores, "paired_comparisons": comparisons,
              "common_match_metrics": common_scores, "model_coverage": coverage,
              "test_predictions": sum(r["partition"] == "test" for r in forecasts),
              "test_matches": len({r["match_id"] for r in forecasts if r["partition"] == "test"}),
              "selection_policy": "models_and_parameters_fixed_before_test; no blending/calibration fitted",
              "coverage_warning": "Per-model coverage can differ; use paired common-match comparisons."}
    (output / "predictions.jsonl").write_text("".join(canonical_json(r) + "\n" for r in forecasts), encoding="utf-8")
    (output / "report.json").write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
