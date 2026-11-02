"""Verify and replay a saved open-information run without network or labels."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl

from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.state import parse_time
from football_research.evaluation.frozen_period import load_frozen_models
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .inquiry_tools import OpenWebTools
from .learned_voi_runtime import LearnedVOIRuntime, digest, _save, VERSION, EVIDENCE_VERSION, TEXT_VERSION, CONTEXT_VERSION, EVIDENCE_ONLY_VERSION, OFFICIAL_CONTEXT_VERSION
from .learned_voi_web import MatchDocumentCandidates


def replay_saved_run(folder, *, output, project_root, base_model=None):
    """Replay raw returns, optionally recovering a pinned synthetic base model.

    ``base_model`` is only for the old constructed-mechanism archives whose
    base model was not persisted. It is a SHA-bound JSON artifact, never an
    injected object/callable. Real prospective archives cannot use this path.
    The entire restored policy must still equal the original manifest below.
    """
    folder, output, root = Path(folder), Path(output), Path(project_root)
    original = json.loads((folder/"run.json").read_text(encoding="utf-8"))
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    if original["configuration_digest"] != digest(config):
        raise ValueError("Run configuration digest changed")
    metadata = config["supervisor_metadata"]
    manifest = config["policy_manifest"]
    base_manifest = manifest["base_policy"] if config["schema"] in {TEXT_VERSION, CONTEXT_VERSION} else manifest
    if config['schema'] == OFFICIAL_CONTEXT_VERSION:
        if base_model is not None:
            raise ValueError('An official context replay cannot override its archived models')
        from .official_context_policy import OfficialContextVOIPolicy
        from football_research.data_adapters.official_context_collection import verify_official_context_run
        verified = verify_official_context_run(folder, config)
        predictor_path = folder/'context-predictor.json'
        value_path = folder/'context-value-model.json'
        policy = OfficialContextVOIPolicy(verified['identity'], verified['predictor'],
            value_model=verified['value_model'], predictor_artifact=predictor_path.read_bytes(),
            value_artifact=value_path.read_bytes() if verified['value_model'] is not None else None,
            lambda_cost=manifest['lambda_cost'],
            **({'role_identity':verified['role_identity']} if 'role_identity' in verified else {}))
    elif config['schema'] == EVIDENCE_ONLY_VERSION:
        if base_model is not None:
            raise ValueError('Evidence-only replay has no probability model to override')
        from .evidence_only_policy import EvidenceOnlyPolicy
        from football_research.data_adapters.evidence_collection import verify_evidence_run
        policy = EvidenceOnlyPolicy.from_manifest(manifest)
        verify_evidence_run(folder, config)
    elif base_manifest.get("schema") == "prospective-market-prior-policy-v1":
        if base_model is not None:
            raise ValueError("A recovered synthetic base model cannot override a real prospective run")
        from football_research.data_adapters.prospective_text import MarketPriorPolicy, verify_prior_archive, verify_run_identity
        policy = MarketPriorPolicy.from_manifest(base_manifest)
        verify_prior_archive(base_manifest, folder/"prior-archive")
        verify_run_identity(folder, config)
    else:
        freeze_path = root/"var/parallel/agent/round175/frozen-next-period.json"
        if sha256(freeze_path.read_bytes()).hexdigest() != metadata["input_bindings"]["frozen_model_config_sha256"]:
            raise ValueError("Frozen model configuration differs from run")
        if base_model is None:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            _, model = load_frozen_models(root, freeze, design_path=root/"docs/research/acquisition/round175-design.md")
        else:
            if (not isinstance(base_model, dict) or set(base_model) != {"path", "sha256"}
                    or config.get("provider_version") != "synthetic-archive-discovery"
                    or config.get("text_response_model", {}).get("evidence_origin") != "constructed_mechanism"):
                raise ValueError("Recovered base artifacts are restricted to the archived synthetic discovery mechanism")
            recovered_path = Path(base_model["path"])
            if sha256(recovered_path.read_bytes()).hexdigest() != base_model["sha256"]:
                raise ValueError("Recovered base model SHA-256 differs")
            from football_research.acquisition.joint_returns import JointReturnPredictor
            model = JointReturnPredictor.load(recovered_path)
        policy = LearnedJointVOIPolicy(model, lambda_cost=base_manifest["lambda_cost"], loss=base_manifest["loss"],
            min_effective_matches=base_manifest["min_effective_matches"], min_class_weight=base_manifest["min_class_weight"])
    if config["schema"] == TEXT_VERSION:
        from football_research.acquisition.text_history import load_text_model
        from .text_voi_policy import TextVOIPolicy
        binding = config["text_response_model"]
        if binding["file"] != "text-response-model.json":
            raise ValueError("Text replay requires its locally archived frozen model")
        model_path = folder/binding["file"]
        if sha256(model_path.read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError("Archived text response model differs from the pre-query binding")
        text_model = load_text_model(model_path)
        if (text_model.model_version != binding["model_version"] or
                text_model.metadata["evidence_origin"] != binding["evidence_origin"]):
            raise ValueError("Text response model identity changed")
        discovery_model = None
        if "discovery_response_model" in config:
            from football_research.acquisition.discovery_returns import DiscoveryReturnModel
            discovery_binding = config["discovery_response_model"]
            if discovery_binding["file"] != "discovery-response-model.json":
                raise ValueError("Discovery replay requires its locally archived frozen model")
            discovery_path = folder/discovery_binding["file"]
            if sha256(discovery_path.read_bytes()).hexdigest() != discovery_binding["sha256"]:
                raise ValueError("Archived discovery model differs from the pre-query binding")
            discovery_model = DiscoveryReturnModel.load(discovery_path, text_model=text_model)
            if (discovery_model.model_version != discovery_binding["model_version"] or
                    discovery_model.metadata["evidence_origin"] != discovery_binding["evidence_origin"]):
                raise ValueError("Discovery response model identity changed")
        risk_model = None
        if "text_risk_model" in config:
            from football_research.acquisition.neighborhood_text_risk import load_text_risk_model
            risk_binding = config["text_risk_model"]
            if risk_binding["file"] != "text-risk-model.json":
                raise ValueError("Empirical risk replay requires its locally archived frozen model")
            risk_path = folder/risk_binding["file"]
            if sha256(risk_path.read_bytes()).hexdigest() != risk_binding["sha256"]:
                raise ValueError("Archived empirical risk model differs from the pre-query binding")
            risk_model = load_text_risk_model(risk_path, text_model=text_model)
            if (risk_model.model_version != risk_binding["model_version"] or
                    risk_model.manifest["evidence_origin"] != risk_binding["evidence_origin"]):
                raise ValueError("Empirical risk model identity changed")
        policy = TextVOIPolicy(policy, text_model, discovery_model=discovery_model, risk_model=risk_model)
    if config["schema"] == CONTEXT_VERSION:
        def bound_model(key, filename):
            binding = config[key]
            if binding["file"] != filename:
                raise ValueError("Context replay requires its locally archived model")
            path = folder/filename
            if sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
                raise ValueError("Context model differs from its pre-query SHA binding")
            return path, binding
        predictor_path, binding = bound_model("context_predictor", "context-predictor.json")
        synthetic_prior = None
        prior_binding = config.get("synthetic_prior")
        if prior_binding is not None:
            if not isinstance(prior_binding, dict) or prior_binding.get("file") != "synthetic-prior-model.json":
                raise ValueError("Synthetic prior replay requires its locally archived model")
            prior_path = folder/prior_binding["file"]
            if sha256(prior_path.read_bytes()).hexdigest() != prior_binding.get("sha256"):
                raise ValueError("Synthetic prior differs from its pre-query SHA binding")
            if prior_binding.get("adapter_schema") == "footballmind-g-synthetic-teaching-prior-adapter-v1":
                from .synthetic_g_teaching_prior import SyntheticGTeachingPriorAdapter
                synthetic_prior = SyntheticGTeachingPriorAdapter.from_path(
                    prior_path, declared_source_path=prior_binding.get("source_path"))
            elif prior_binding.get("adapter_schema") == "footballmind-g-prior-adapter-v1":
                from .g_teacher_prior import SyntheticGTeacherPriorAdapter
                synthetic_prior = SyntheticGTeacherPriorAdapter.from_path(
                    prior_path, declared_source_path=prior_binding.get("source_path"))
            else:
                raise ValueError("Unsupported synthetic prior adapter schema")
            if synthetic_prior.manifest != manifest.get("synthetic_prior"):
                raise ValueError("Synthetic prior policy manifest differs from its archived binding")
        elif "synthetic_prior" in manifest:
            raise ValueError("Policy declares a synthetic prior without an archived binding")
        if manifest.get("schema") in {"entity-semantic-market-shadow-f-only-policy-v1",
                                      "entity-semantic-market-shadow-f-only-policy-v2",
                                      "entity-semantic-market-shadow-f-only-policy-v3",
                                      "entity-semantic-market-shadow-f-only-policy-v4"}:
            from football_research.acquisition.entity_semantic_f import EntitySemanticF
            from .entity_semantic_f_only_policy import EntitySemanticFOnlyPolicy, TRACE_FILE
            if "context_value_model" in config or synthetic_prior is not None:
                raise ValueError("Entity shadow replay cannot enable g or synthetic selection")
            trace_binding = original.get("semantic_extraction_archive", {})
            trace_path = folder / TRACE_FILE
            if (trace_binding.get("file") != TRACE_FILE
                    or sha256(trace_path.read_bytes()).hexdigest() != trace_binding.get("sha256")):
                raise ValueError("Entity semantic extraction trace differs from its saved SHA binding")
            model = EntitySemanticF.load(predictor_path)
            if (model.model_version != binding["model_version"] or
                    model.metadata["evidence_origin"] != binding["evidence_origin"] or
                    binding["sha256"] != manifest.get("predictor_sha256")):
                raise ValueError("Entity shadow model differs from its frozen binding")
            policy = EntitySemanticFOnlyPolicy(policy, model, fixture=manifest["fixture"],
                artifact=predictor_path.read_bytes(), semantic_config=manifest["semantic_config"],
                replay_trace=json.loads(trace_path.read_bytes()),
                operational_body_admission="body_admission" in manifest["prediction_pipeline_manifest"])
        elif manifest.get("schema") == "historical-f-only-research-policy-v1":
            from football_research.acquisition.historical_context_training import HistoricalContextModel
            from .historical_f_only_policy import HistoricalFOnlyPolicy
            if "context_value_model" in config or synthetic_prior is not None:
                raise ValueError("f-only replay cannot enable a bundled gain model or synthetic prior")
            model = HistoricalContextModel.load(predictor_path)
            if (model.model_version != binding["model_version"] or
                    model.metadata["evidence_origin"] != binding["evidence_origin"] or
                    binding["sha256"] != manifest.get("predictor_sha256")):
                raise ValueError("f-only model differs from its frozen prediction binding")
            policy = HistoricalFOnlyPolicy(policy, model, fixture=manifest["fixture"],
                                          artifact=predictor_path.read_bytes())
        elif manifest.get("schema") == "historical-context-research-policy-v1":
            from football_research.acquisition.historical_context_training import HistoricalContextModel
            from .historical_context_policy import HistoricalContextVOIPolicy
            if "context_value_model" not in config:
                raise ValueError("Historical context replay requires the combined f/g snapshot")
            value_path, value_binding = bound_model("context_value_model", "context-value-model.json")
            artifact = predictor_path.read_bytes()
            if artifact != value_path.read_bytes():
                raise ValueError("Historical context predictor and value snapshots must be identical")
            if manifest.get("predictor_sha256") != binding["sha256"]:
                raise ValueError("Historical context policy differs from its archived model binding")
            model = HistoricalContextModel.load(predictor_path)
            for model_binding in (binding, value_binding):
                if (model.model_version != model_binding["model_version"] or
                        model.metadata["evidence_origin"] != model_binding["evidence_origin"]):
                    raise ValueError("Historical context combined model identity changed")
            policy = HistoricalContextVOIPolicy(policy, model, fixture=manifest["fixture"], artifact=artifact,
                state_generalization=manifest.get("state_generalization", False),
                synthetic_prior=synthetic_prior)
        else:
            from football_research.acquisition.context_predictor import ContextPredictor
            from .context_voi_policy import ContextVOIPolicy
            predictor = ContextPredictor.load(predictor_path)
            if (predictor.model_version != binding["model_version"] or
                    predictor.metadata["evidence_origin"] != binding["evidence_origin"]):
                raise ValueError("Context predictor identity changed")
            value_model = None
            if "context_value_model" in config:
                from football_research.acquisition.context_value import ContextValueModel
                value_path, value_binding = bound_model("context_value_model", "context-value-model.json")
                value_model = ContextValueModel.load(value_path, predictor_path=predictor_path)
                if (value_model.model_version != value_binding["model_version"] or
                        value_model.metadata["evidence_origin"] != value_binding["evidence_origin"]):
                    raise ValueError("Context value model identity changed")
            policy = ContextVOIPolicy(policy, predictor, physical_match_id=manifest["physical_match_id"],
                                      value_model=value_model, predictor_artifact=predictor_path.read_bytes(),
                                      value_artifact=value_path.read_bytes() if value_model is not None else None,
                                      synthetic_prior=synthetic_prior)
    if policy.manifest != manifest or policy.version != config["policy_version"]:
        raise ValueError("Frozen policy differs from run")
    if "learning_cohort" in metadata:
        from football_research.data_adapters.prospective_text import verify_run_learning
        verify_run_learning(folder, config)
    if config["schema"] == CONTEXT_VERSION and "context_calibration" in metadata:
        from football_research.data_adapters.context_collection import verify_context_run
        verify_context_run(folder, config)
    if config["schema"] == CONTEXT_VERSION and "context_confirmation" in metadata:
        from football_research.data_adapters.context_collection import verify_context_confirmation_run
        verify_context_confirmation_run(folder, config)
    gen = metadata["generator"]
    base_gen = deepcopy(gen)
    shadow_config = base_gen.pop("observable_prequery_shadow", None)
    if base_gen["version"] == MatchDocumentCandidates.version:
        generator = MatchDocumentCandidates(match_terms=base_gen["match_terms"], seed_urls=base_gen["seed_urls"],
            allowed_hosts=base_gen["allowed_hosts"], numeric_actions=base_gen["numeric_actions"],
            web_cost=base_gen["web_cost"], max_discovered_links=base_gen["max_discovered_links"])
    else:
        from .evidence_candidates import EvidenceDocumentCandidates
        generator = EvidenceDocumentCandidates.from_configuration(base_gen)
    if shadow_config is not None:
        from .observable_prequery_shadow import ObservablePrequeryShadow, MODEL_FILE
        generator = ObservablePrequeryShadow.from_configuration(generator, shadow_config, folder / MODEL_FILE)
    if generator.configuration() != gen:
        raise ValueError("Candidate generation configuration changed")
    receipts, observations = {}, set()
    # Verify projected content from raw bytes, not merely a digest declared by a receipt.
    archive = RawArchive(folder/"source-archive")
    if "news_discovery" in metadata["transport"]:
        from .news_discovery import NewsDiscoveryTools
        NewsDiscoveryTools.from_configuration(archive, metadata["transport"])
    elif "source_route_policy" in metadata["transport"]:
        from .source_recovery import SourceRoutedOpenWebTools
        SourceRoutedOpenWebTools.from_configuration(archive, metadata["transport"])
    version = int(metadata["transport"]["version"].rsplit("v", 1)[1])
    for cycle in original["cycles"]:
        step = cycle["step"]
        persisted = json.loads((folder/f"cycle-{step:03}.json").read_text(encoding="utf-8"))
        if persisted != cycle:
            raise ValueError("Per-cycle record differs from completed run")
        if cycle["selected_action"] is None:
            continue
        if cycle["execution_status"] != "returned":
            raise ValueError("Only fully reconciled returns can be replayed")
        returned = json.loads((folder/f"receipt-{step:03}.json").read_text(encoding="utf-8"))
        if "observation" in returned.get("receipt", {}):
            record = returned["receipt"]
            obs = Observation.model_validate(record["observation"])
            if obs.id in observations:
                raise ValueError("A source observation was billed more than once")
            observations.add(obs.id)
            doc = returned["document"]
            if "news_discovery" in metadata["transport"]:
                tools = NewsDiscoveryTools.from_configuration(archive, metadata["transport"],
                    clock=lambda d=doc: parse_time(d["available_at"]))
            else:
                tools = OpenWebTools(archive, gen["allowed_hosts"], extraction_version=version,
                                     clock=lambda d=doc: parse_time(d["available_at"]))
            projected = tools.project(obs, record["prepared"])
            def url_identity(url):
                parts = urlsplit(url)
                return (parts.scheme, parts.netloc, parts.path,
                        tuple(sorted(parse_qsl(parts.query, keep_blank_values=True))), parts.fragment)
            # Legacy v1/v2 URL spelling follows dictionary insertion order;
            # JSON persistence may reorder keys. No other field is normalized.
            if version < 3 and url_identity(projected["url"]) == url_identity(doc["url"]):
                projected["url"] = doc["url"]
            if projected != doc:
                raise ValueError("Document projection does not reproduce archived response bytes")
            if returned["actual_cost"]["requests"] != 1:
                raise ValueError("Archived one-attempt request accounting differs")
        receipts[step] = returned
    # A prospective run may have archived each pre-I/O candidate menu in the
    # same immutable RawArchive as its response. Verify the referenced payload
    # exists before replay, then return the original reference so replay never
    # creates a second raw record or changes the evidence trail.
    menu_refs = {}
    if config.get("trajectory_raw_archive"):
        by_path = {obs.payload_path: obs for obs in archive.observations("agent_trajectory")}
        for cycle in original["cycles"]:
            path = cycle.get("menu_raw_path")
            if not path:
                continue
            obs = by_path.get(path)
            if obs is None or (cycle.get("menu_raw_sha256") and obs.content_hash != cycle["menu_raw_sha256"]):
                raise ValueError("Archived candidate menu reference is missing or has a different digest")
            archive.read_body(obs)
            menu_refs[cycle["step"]] = {k: cycle[k] for k in
                ("menu_raw_path", "menu_raw_sha256", "menu_raw_observation_id") if k in cycle}

    def replay_menu(payload):
        # Runtime still reconstructs the public menu deterministically; this
        # sidecar only reuses its immutable archive reference.
        return deepcopy(menu_refs.get(payload["step"]))
    context_refs = []
    if config.get("initial_state_timing"):
        context_refs.append(config["initial_state_timing"])
        for cycle in original["cycles"]:
            context_refs.append(cycle["before_state_timing"])
            if (cycle.get("after_state_timing") or {}).get("raw_path"):
                context_refs.append(cycle["after_state_timing"])
        context_refs.append(original["final_state_timing"])
    context_payloads = {}
    by_path = {obs.payload_path: obs for obs in archive.observations("agent_trajectory")}
    for ref in context_refs:
        obs = by_path.get(ref["raw_path"])
        if obs is None or obs.content_hash != ref["raw_sha256"] or obs.id != ref["raw_observation_id"]:
            raise ValueError("Archived context reference is missing or has a different digest")
        context_payloads[ref["raw_path"]] = json.loads(archive.read_body(obs))
    context_index = 0

    def replay_context(payload):
        nonlocal context_index
        ref = context_refs[context_index]
        context_index += 1
        if payload != context_payloads[ref["raw_path"]]:
            raise ValueError("Runtime state handoff does not reproduce the archived context")
        return deepcopy(ref)
    count = 0

    def execute(action, public):
        nonlocal count
        cycle = original["cycles"][count]
        count += 1
        if action != cycle["selected_action"] or digest(public) != cycle["public_digest"]:
            raise ValueError("Replay selected a different action or queried from a different state")
        return deepcopy(receipts[cycle["step"]])

    trajectory = config.get("trajectory_schema")
    runtime = LearnedVOIRuntime(policy, seed=config["inquiry_exploration"]["seed"],
        exploration_limits=config["exploration_limits"], max_exploration_actions=config["max_exploration_actions"],
        max_steps=config["max_steps"],
        policy_arm=config.get("policy_arm") if trajectory else None,
        policy_arm_version=config.get("policy_arm_version", "footballmind-policy-arm-v1"),
        arm_assigned_at=config.get("arm_assigned_at"),
        random_seed=config.get("random_seed", config["inquiry_exploration"]["seed"]),
        propensity_logging=config.get("propensity_logging", True),
        legacy_receipt_clock=bool(trajectory) and not any("receipt_timing_basis" in c for c in original["cycles"]),
        legacy_predictor_context=config.get("predictor_context_schema") not in
            {"runtime-predictor-context-v3", "runtime-predictor-context-v4"})
    projection = config.get("evidence_projection")
    if projection:
        if config["schema"] not in {EVIDENCE_VERSION, TEXT_VERSION, CONTEXT_VERSION, EVIDENCE_ONLY_VERSION, OFFICIAL_CONTEXT_VERSION}:
            raise ValueError("Unsupported evidence runtime version")
        ticks = [config["created_at"]]
        if context_refs:
            ticks.append(config["initial_state_timing"]["observed_at"])
        for cycle in original["cycles"]:
            if context_refs and not getattr(policy, "predictor_only", False):
                ticks.append(cycle["before_state_timing"]["observed_at"])
            ticks.append(cycle["public_before"].get("prediction_input_at", cycle["public_before"]["decision_at"]))
            if context_refs and getattr(policy, "predictor_only", False):
                ticks.append(cycle["before_state_timing"]["observed_at"])
            # Every projected trajectory with a selected action consumes a
            # local return clock.  Restricting this to the newer text/context
            # schema constants shifts v2 replay timestamps and can change the
            # next public state, even though the archived response is intact.
            if cycle["selected_action"] is not None and "returned_at" in cycle:
                ticks.append(cycle["returned_at"])
            if context_refs and (cycle.get("after_state_timing") or {}).get("raw_path"):
                ticks.append(cycle["after_state_timing"]["observed_at"])
        ticks.append(original["finished_at"])
        if context_refs:
            ticks.append(original["final_state_timing"]["observed_at"])
        times = iter(ticks)
        clock = lambda: parse_time(next(times))
    else:
        if config["schema"] != VERSION:
            raise ValueError("Unsupported runtime version")
        clock = lambda: parse_time(config["created_at"])
    replayed = runtime.run(config["initial_state"], run_id=config["run_id"], limits=config["limits"],
        propose=generator, execute=execute, output=output, provider_version=config["provider_version"],
        generator_version=config["generator_version"], clock=clock,
        run_metadata=metadata, evidence_match=projection["match"] if projection else None,
        evidence_version=projection["version"] if projection else None,
        trajectory_recorder=replay_menu if menu_refs else None,
        context_recorder=replay_context if context_refs else None)
    if replayed != original:
        raise ValueError("Full replay differs, including decisions, reservations or posterior updates")
    if projection:
        from .match_analysis_report import write_analysis_report
        write_analysis_report(replayed, match=projection["match"], output=output)
    examples = []
    for cycle in original["cycles"]:
        action = cycle["selected_action"]
        if action is None:
            continue
        returned = receipts[cycle["step"]]
        exploration = cycle["exploration_record"]
        value = next(c["value"] for c in cycle["strict_decision"]["candidates"] if c["action_id"] == action["action_id"])
        examples.append({"schema": "open-information-learning-record-v1", "run_id": original["run_id"],
            "match_id": config["initial_state"]["match_id"], "step": cycle["step"],
            "public_before_digest": cycle["public_digest"], "action": action,
            "generated_candidate_ids": [a["action_id"] for a in cycle["generated_actions"]],
            "selection_basis": cycle["selection_basis"],
            "conditional_selection_probability": exploration["selected_p"] if exploration else None,
            "exploration_record": exploration, "estimated_net_voi": value["net_voi"],
            "response_status": returned["status"], "admission": cycle["admission"],
            "document_id": returned.get("document", {}).get("document_id"),
            "raw_sha256": returned.get("document", {}).get("raw_sha256"),
            "observed_at": returned.get("document", {}).get("observed_at"),
            "available_at": returned.get("document", {}).get("available_at"),
            "prediction_before": cycle["prediction_before"], "prediction_after": cycle["prediction_after"],
            "declared_reservation": cycle["reserved_cost"], "actual_cost": returned["actual_cost"],
            "outcome": None, "realized_predictive_gain": None, "numeric_voi_training_eligible": False,
            "response_learning_status": "raw_return_and_selection_logged_outcome_and_response_mapping_pending",
            "scope": "conditional_on_generated_menu_not_open_world_propensity"})
        if config["schema"] == TEXT_VERSION:
            examples[-1].update(
                text_model_version=config["text_response_model"]["model_version"],
                text_model_origin=config["text_response_model"]["evidence_origin"],
                response_update=cycle["update"],
                response_learning_status="frozen_history_conditional_text_model_unvalidated")
            if "discovery_response_model" in config:
                examples[-1].update(
                    discovery_model_version=config["discovery_response_model"]["model_version"],
                    planning_horizon=2,
                    value_scope="model_conditional_bounded_two_step")
            if "text_risk_model" in config:
                examples[-1].update(
                    risk_model_version=config["text_risk_model"]["model_version"],
                    value_scope="empirical_fixed_predictor_log_loss_reduction")
    _save(output/"learning-records.json", examples)
    report = {"schema": "open-voi-replay-acceptance-v1", "status": "passed",
              "verified_at": datetime.now(timezone.utc).isoformat(), "run_id": original["run_id"],
              "identical_replay": True, "cycles": len(original["cycles"]),
              "returns": len(receipts), "raw_responses_reprojected": len(observations),
              "replay_network_requests": 0, "learning_records": len(examples),
              "learning_records_sha256": sha256((output/"learning-records.json").read_bytes()).hexdigest(),
              "original_run_sha256": sha256((folder/"run.json").read_bytes()).hexdigest(),
              "posterior_benefit_proven": False, "numeric_voi_training_eligible": False}
    if base_model is not None:
        report["recovered_synthetic_base_model"] = dict(base_model)
        report["scope"] = "synthetic_mechanism_only"
    _save(output/"acceptance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(replay_saved_run(args.run, output=args.output, project_root=Path(__file__).resolve().parents[1]), indent=2))


if __name__ == "__main__":
    main()
