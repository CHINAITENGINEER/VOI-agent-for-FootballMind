"""Headless learned-VoI acquisition for one upcoming exported fixture.

The only match-specific user input is its identity (or exact 'home vs away').
Source capabilities, frozen models and budgets are deployment configuration.
The existing model retains its historical-reconstruction scope. Current web
observations are collected before the declared forecast cutoff; this does not
establish prospective probability calibration or a verified fixture/result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from football_research.acquisition.data import ControlledEpisode
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.state import GROUP_FIELDS, parse_time
from football_research.data_adapters.platform_history import verify_completed_export, load_completed_history_export
from football_research.data_adapters.batch_features import FeatureIndex
from football_research.data_adapters.types import json_record
from football_research.evaluation.frozen_period import load_frozen_models
from football_spider.storage.raw_archive import RawArchive
from .inquiry_transport import RoutedOpenWebTools
from .learned_voi_runtime import LearnedVOIRuntime, digest, _save
from .learned_voi_web import MatchDocumentCandidates, ArchivedAndWebExecutor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT/"var/parallel/collectors/round175/exports/understat-2026-historical_reconstruction-c6bb4e8c81a6f34d"


def prepare_fixture(bundle, *, match=None, now=None):
    """Use exact verified identities; no label-based or model-gain case selection."""
    now = parse_time(now or datetime.now(timezone.utc))
    verified = verify_completed_export(bundle)
    if verified.manifest.get("source") != "understat" or str(verified.manifest.get("season")) != "2026":
        raise ValueError("This source-capability preset is bound to the Understat 2026 export")
    leagues = {entry.get("original_context", {}).get("league")
               for entry in verified.manifest.get("source_evidence", [])}
    if leagues != {"EPL"}:
        raise ValueError("The EPL source preset requires an unambiguous EPL source lineage")
    data = load_completed_history_export(bundle)
    if verify_completed_export(bundle).manifest_sha256 != verified.manifest_sha256:
        raise ValueError("Export changed while preparing fixture")
    if parse_time(verified.manifest["source_observed_at"]) > now:
        raise ValueError("Fixture export is not observed yet")
    raw_by_id = {r["match_id"]: r for r in verified.rows("matches")}
    if len(raw_by_id) != len(data.matches):
        raise ValueError("Duplicate exported match identities")
    eligible = [m for m in data.matches if m.status == "scheduled" and m.outcome is None
                and now < m.kickoff-timedelta(hours=24)]
    eligible.sort(key=lambda m: (m.kickoff, m.match_id))
    if match is not None:
        target = match.strip().casefold()
        eligible = [m for m in eligible if target in {m.match_id.casefold(),
                    (raw_by_id[m.match_id]["home_team_name"]+" vs "+raw_by_id[m.match_id]["away_team_name"]).casefold()}]
        if len(eligible) != 1:
            raise ValueError("Match must uniquely identify an upcoming exported fixture beyond its 24h cutoff")
    if not eligible:
        raise ValueError("No upcoming fixture with a future 24h forecast cutoff")
    selected = eligible[0]
    snapshots = verified.rows("features")
    versions = {row.get("feature_version") for row in snapshots}
    if len(versions) != 1 or None in versions:
        raise ValueError("One declared feature version is required")
    version = next(iter(versions))
    index = FeatureIndex.from_rows(snapshots, allowed_features=[f for fs in GROUP_FIELDS.values() for f in fs])
    cutoff = selected.kickoff-timedelta(hours=24)
    joined = index.join(selected, prediction_cutoff=cutoff, information_mode=selected.information_mode,
                        feature_version=version, allow_missing_values=True)
    if selected.fixture_available_at is None or selected.fixture_available_at > now or joined["features_available_at"] > now:
        raise ValueError("Initial archived fixture/features are not available at the actual collection time")
    row = json_record(selected)
    # The environment needs no evaluator label, even though the source has nulls.
    for name in ("home_goals", "away_goals", "outcome", "result_available_at"):
        row.pop(name, None)
    row.update({f: joined.get(f) for fs in GROUP_FIELDS.values() for f in fs})
    row.update(prediction_cutoff=cutoff.isoformat(), features_available_at=joined["features_available_at"].isoformat(),
               feature_version=version, dataset_version=data.dataset_version)
    original = raw_by_id[selected.match_id]
    names = [original["home_team_name"], original["away_team_name"]]
    report = {"match_id": selected.match_id, "match": " vs ".join(names),
              "kickoff": selected.kickoff.isoformat(), "prediction_cutoff": cutoff.isoformat(),
              "prepared_at": now.isoformat(), "export_manifest_sha256": verified.manifest_sha256,
              "source_observed_at": verified.manifest["source_observed_at"],
              "case_selection": "exact_user_match" if match else "earliest_upcoming_then_match_id",
              "fixture_identity_scope": "existing_single_provider_export_not_independent_verification",
              "model_information_mode": selected.information_mode,
              "cutoff_semantics": "fixed_future_admission_horizon_not_the_actual_query_clock",
              "label_available": False, "historical_feature_assumptions_retained": True}
    return row, names, report


def launch(*, bundle, output, freeze_path, match=None, route="environment", seed=176,
           exploration_actions=3, source_profile="club-news", source_routes=None,
           text_model_path=None, discovery_model_path=None, text_risk_model_path=None,
           evidence_version=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Run directory exists; verify its receipts instead of repeating live requests")
    from .match_evidence import SUPPORTED_VERSIONS
    if evidence_version is not None and evidence_version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported match evidence version")
    if evidence_version is not None and source_profile != "club-news":
        raise ValueError("An explicit evidence projection requires the club-news source profile")
    if discovery_model_path is not None and text_model_path is None:
        raise ValueError("A discovery model requires its jointly bound text response model")
    if text_risk_model_path is not None and (text_model_path is None or discovery_model_path is not None):
        raise ValueError("An empirical text risk model requires --text-model and cannot combine with discovery")
    if text_model_path is not None and source_profile != "club-news":
        raise ValueError("A text response model requires the club-news source capability profile")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    _, model = load_frozen_models(ROOT, freeze, design_path=ROOT/"docs/research/acquisition/round175-design.md")
    row, names, fixture_report = prepare_fixture(bundle, match=match)
    text_model, discovery_model, risk_model = None, None, None
    numeric_actions = freeze["actions"]
    if text_model_path is not None:
        from football_research.acquisition.text_history import load_text_model
        text_model = load_text_model(text_model_path)
        if text_model.metadata.get("evidence_origin") != "real_observational":
            raise ValueError("Live text acquisition requires a real_observational text model; constructed models are offline only")
        if row.get("information_mode") != "operational":
            raise ValueError("Live text acquisition cannot mix historical_reconstruction initialization with operational text training")
        numeric_actions = []
        if discovery_model_path is not None:
            from football_research.acquisition.discovery_returns import DiscoveryReturnModel
            discovery_model = DiscoveryReturnModel.load(discovery_model_path, text_model=text_model)
            if discovery_model.metadata.get("evidence_origin") != "real_observational":
                raise ValueError("Live discovery acquisition requires a real_observational model")
        if text_risk_model_path is not None:
            from football_research.acquisition.neighborhood_text_risk import load_text_risk_model
            risk_model = load_text_risk_model(text_risk_model_path, text_model=text_model)
            if risk_model.manifest.get("evidence_origin") != "real_observational":
                raise ValueError("Live empirical text acquisition requires real_observational supervision")
    # Only free fixture context reaches the model before acquisition.
    initial = ControlledEpisode(row, numeric_actions).visible_state()
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": freeze["lambda_money"]})
    if text_model is not None:
        from .text_voi_policy import TextVOIPolicy
        policy = (TextVOIPolicy(policy, text_model, discovery_model=discovery_model)
                  if discovery_model is not None else TextVOIPolicy(policy, text_model))
        if risk_model is not None:
            policy = TextVOIPolicy(policy.base_policy, text_model, risk_model=risk_model)
    if type(exploration_actions) is not int or not 0 <= exploration_actions <= 8:
        raise ValueError("At most eight public-source attempts per run")
    identity = {"match_id": row["match_id"], "home_team": names[0], "away_team": names[1],
                "kickoff_at": fixture_report["kickoff"], "prediction_cutoff": fixture_report["prediction_cutoff"]}
    evidence_match = None
    if source_profile == "club-news":
        from .match_source_profiles import resolve_profile
        from .evidence_candidates import EvidenceDocumentCandidates
        profile = resolve_profile(*names)
        if profile["status"] == "unsupported":
            raise ValueError("No registered public source capability for this pair of teams; select a supported fixture")
        for team in profile["teams"]:
            if team["canonical_name"]:
                identity[team["side"]+"_aliases"] = [team["canonical_name"]]
        seeds, hosts = profile["seed_urls"], profile["allowed_hosts"]
        generator = EvidenceDocumentCandidates(match_terms=names, seed_urls=seeds, allowed_hosts=hosts,
            numeric_actions=numeric_actions, web_cost={"money": .01, "requests": 1.},
            match=identity, profile=profile)
        evidence_match = identity
    elif source_profile == "understat":
        # Retain the original generator for reproducing earlier runs.
        seeds, hosts = ["https://understat.com/league/EPL/2026"], ["understat.com"]
        generator = MatchDocumentCandidates(match_terms=names, seed_urls=seeds, allowed_hosts=hosts,
            numeric_actions=numeric_actions, web_cost={"money": .01, "requests": 1.})
    else:
        raise ValueError("Unknown source capability profile")
    if source_routes:
        from .source_recovery import SourceRoutedOpenWebTools
        tools = SourceRoutedOpenWebTools(RawArchive(output/"source-archive"), hosts,
            default_route=route, route_overrides=source_routes, extraction_version=3)
    else:
        tools = RoutedOpenWebTools(RawArchive(output/"source-archive"), hosts, route=route, extraction_version=3)
    executor = ArchivedAndWebExecutor(row, numeric_actions, tools, seed_urls=seeds)
    run_id = "acquisition-"+digest({"fixture": fixture_report, "seed": seed, "output": str(output.resolve())})[:24]
    executor.run_id = run_id
    exploration_limits = {"money": .01*exploration_actions, "requests": exploration_actions}
    limits = {"money": freeze["limits"]["money"]+exploration_limits["money"],
              "requests": freeze["limits"]["requests"]+exploration_actions}
    runtime = LearnedVOIRuntime(policy, seed=seed, exploration_limits=exploration_limits,
                                 max_exploration_actions=exploration_actions)
    # Bind the full configuration before I/O, including source catalog and route.
    provider_version = executor.version+":"+digest(tools.configuration())
    generator_version = generator.version+":"+digest(generator.configuration())
    metadata = {"fixture": fixture_report, "generator": generator.configuration(),
                "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(Path(freeze_path).read_bytes()).hexdigest(),
                    "model_version": model.model_version, "export_manifest_sha256": fixture_report["export_manifest_sha256"],
                    "hyperparameters_changed": False, "case_selected_by_outcome_or_query_value": False}}
    if text_model is not None:
        metadata["input_bindings"].update(
            text_response_model_version=text_model.model_version,
            text_response_model_sha256=sha256(Path(text_model_path).read_bytes()).hexdigest(),
            text_response_model_evidence_origin=text_model.metadata["evidence_origin"],
            numeric_acquisition_disabled=True,
            initialization_mode=initial["information_mode"])
        if discovery_model is not None:
            metadata["input_bindings"].update(
                discovery_model_version=discovery_model.model_version,
                discovery_model_sha256=sha256(Path(discovery_model_path).read_bytes()).hexdigest(),
                planning_horizon=2)
        if risk_model is not None:
            metadata["input_bindings"].update(
                text_risk_model_version=risk_model.model_version,
                text_risk_model_sha256=sha256(Path(text_risk_model_path).read_bytes()).hexdigest(),
                value_scope="empirical_fixed_predictor_log_loss_reduction")
    result = runtime.run(initial, run_id=run_id, limits=limits, propose=generator, execute=executor,
                         output=output, provider_version=provider_version, generator_version=generator_version,
                         run_metadata=metadata, evidence_match=evidence_match, evidence_version=evidence_version)
    _save(output/"source-configuration.json", {"generator": generator.configuration(), "transport": tools.configuration()})
    _save(output/"fixture.json", fixture_report)
    _save(output/"input-bindings.json", metadata["input_bindings"])
    from .match_analysis_report import write_analysis_report
    analysis = write_analysis_report(result, match=identity, output=output)
    return {"run_id": run_id, "match": fixture_report["match"], "output": str(output.resolve()),
            "attempts": len(result["attempts"]), "exploration_actions": result["exploration_actions"],
            "documents": len(result["documents"]), "actual_http_requests": result["measured_resources"]["requests"],
            "evidence_records": len(result.get("evidence", {}).get("evidence", [])),
            "source_profile": source_profile, "analysis": analysis,
            "terminal_selection": result["terminal_selection"],
            "scope": ("current_open_information_text_response_acquisition_not_a_scored_experiment" if text_model is not None else
                      "current_web_collection_with_frozen_reconstructed_predictor_not_a_scored_experiment"),
            "outcome": None, "probability_benefit_proven": False}


def _source_route(value):
    if not isinstance(value, str) or value.count("=") != 1:
        raise argparse.ArgumentTypeError("Source route must be HOST=environment or HOST=direct")
    host, route = value.split("=", 1)
    host = host.lower()
    if (not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
            or ".." in host or route not in {"environment", "direct"}):
        raise argparse.ArgumentTypeError("Source route requires an exact hostname and environment|direct")
    return host, route


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, help="New run directory; omitted creates a timestamped directory under var/runs")
    parser.add_argument("--freeze", type=Path, default=ROOT/"var/parallel/agent/round175/frozen-next-period.json")
    parser.add_argument("--match", help="Existing match ID or exact 'home vs away'; omitted selects earliest upcoming")
    parser.add_argument("--route", choices=["environment", "direct"], default="environment")
    parser.add_argument("--source-route", action="append", type=_source_route, default=[], metavar="HOST=ROUTE",
                        help="Optional exact source route override; repeat for distinct hosts, one attempt per action")
    parser.add_argument("--text-model", type=Path,
                        help="Frozen real_observational text model; requires operational fixture initialization")
    parser.add_argument("--discovery-model", type=Path,
                        help="Optional learned two-step discovery model bound to --text-model")
    parser.add_argument("--text-risk-model", type=Path,
                        help="Optional real-observation empirical gain calibration bound to --text-model; excludes discovery")
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--exploration-actions", type=int, default=3)
    parser.add_argument("--sources", choices=["club-news", "understat"], default="club-news")
    from .match_evidence import SUPPORTED_VERSIONS
    parser.add_argument("--evidence-version", choices=SUPPORTED_VERSIONS,
                        help="Explicit evidence projection; omitted preserves the existing default")
    args = parser.parse_args(argv)
    source_routes = {}
    for host, source_route in args.source_route:
        if host in source_routes:
            parser.error("--source-route cannot repeat the same hostname")
        source_routes[host] = source_route
    output = args.output or ROOT/"var/runs"/datetime.now(timezone.utc).strftime("match-%Y%m%dT%H%M%S.%fZ")
    report = launch(bundle=args.bundle, output=output, freeze_path=args.freeze, match=args.match,
                    route=args.route, seed=args.seed, exploration_actions=args.exploration_actions,
                    source_profile=args.sources, source_routes=source_routes or None,
                    text_model_path=args.text_model, discovery_model_path=args.discovery_model,
                    text_risk_model_path=args.text_risk_model, evidence_version=args.evidence_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
