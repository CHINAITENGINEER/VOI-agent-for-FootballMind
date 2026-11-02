"""All ready-path examples are invented metadata fixtures, not source evidence."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_agent.prospective_contract import VERSION, VALUE_TARGET, validate_prospective_contract, validate_runtime_metadata


def digest(text):
    return sha256(text.encode()).hexdigest()


def fixture():
    domain = {"sport": "football", "category": "men_senior", "competition_ids": ["fixture-league"],
              "stage_types": ["league"], "entity_namespace": "fixture-provider"}
    hashes = {k: digest("constructed-" + k) for k in ("predictor", "calibrator", "catalog", "lambda", "return-schema")}
    units = {"money": "USD", "requests": "HTTP_request", "latency_ms": "ms", "tokens": "token"}
    weights = {"money": .5, "requests": .01, "latency_ms": 0., "tokens": 0.}
    contract = {"schema": VERSION, "mode": "prospective_operational", "freeze_at": "2026-09-01T00:00:00Z", "domain": domain,
        "deployment": {"start": "2026-09-02T00:00:00Z", "end": "2026-09-20T00:00:00Z", "min_lead_hours": 1., "max_lead_hours": 48.},
        "selection": {"kind": "all_eligible", "time_field": "prediction_cutoff", "competition_ids": ["fixture-league"],
                      "frozen_at": "2026-08-31T00:00:00Z"},
        "artifacts": {k: {"sha256": hashes[k], "version": "constructed-" + k} for k in ("predictor", "calibrator", "catalog")},
        "value_target": VALUE_TARGET, "cost_units": units, "tools": ["results_view"],
        "policy": {"horizon": 1, "risk_level": "mean", "lambda_cost": weights, "reporting_utility_weights": deepcopy(weights),
                   "lambda_source": {"kind": "heldout_temporal_validation", "sha256": hashes["lambda"]}},
        "information_semantics": {"information_mode": "operational", "available_at_rule": "max_observed_ingested_registered",
                                  "pre_action_decision_archive": True, "separate_outcome_store": True}}
    predictor = {**contract["artifacts"]["predictor"], "available_at": "2026-08-02T00:00:00Z", "trained_as_of": "2026-08-01T00:00:00Z",
                 "max_data_available_at": "2026-07-31T00:00:00Z", "training_domain": deepcopy(domain), "information_mode": "operational",
                 "return_schema_sha256": hashes["return-schema"], "action_ids": ["results_view"]}
    calibrator = {**contract["artifacts"]["calibrator"], "available_at": "2026-08-15T00:00:00Z", "trained_as_of": "2026-08-10T00:00:00Z",
                  "max_data_available_at": "2026-08-09T00:00:00Z", "diagnostics_data_available_at": "2026-08-14T00:00:00Z",
                  "training_domain": deepcopy(domain), "information_mode": "operational", "value_target": VALUE_TARGET,
                  "reference_predictor_sha256": hashes["predictor"], "reference_predictor_version": predictor["version"],
                  "reference_catalog_sha256": hashes["catalog"]}
    catalog = {**contract["artifacts"]["catalog"], "available_at": "2026-08-03T00:00:00Z", "information_mode": "operational",
        "return_schema_sha256": hashes["return-schema"], "actions": [{"action_id": "results_view", "source_id": "constructed-source",
            "cost_scope": "real_external_request", "expected_cost": {k: {"amount": 1. if k == "requests" else .02 if k == "money" else 0.,
                "unit": unit, "basis": "known"} for k, unit in units.items()}}]}
    evidence = {"verifier": {"name": "constructed-test-verifier", "version": "fixture-v1", "report_sha256": digest("fixture-proof")},
        "artifacts": {"predictor": predictor, "calibrator": calibrator, "catalog": catalog},
        "lambda_selection": {"sha256": hashes["lambda"], "kind": "heldout_temporal_validation", "available_at": "2026-08-21T00:00:00Z",
            "selected_at": "2026-08-20T00:00:00Z", "max_data_available_at": "2026-08-19T00:00:00Z", "information_mode": "operational",
            "training_domain": deepcopy(domain), "predictor_sha256": hashes["predictor"], "calibrator_sha256": hashes["calibrator"],
            "catalog_sha256": hashes["catalog"], "lambda_cost": deepcopy(weights), "reporting_utility_weights": deepcopy(weights),
            "split_manifest_sha256": digest("fixed-validation-split")},
        "sources": [{"source_id": "constructed-source", "domain": deepcopy(domain), "information_mode": "operational",
            "archive_basis": "observed_operational_archive", "archive_manifest_sha256": digest("fixture-archive"),
            "verified_at": "2026-08-23T00:00:00Z", "records_request_response_costs": True, "records_pre_action_intent": True,
            "probe": {"sha256": digest("fixture-receipt"), "observed_at": "2026-08-22T12:00:00Z", "ingested_at": "2026-08-22T12:00:01Z",
                      "registered_at": "2026-08-22T12:00:02Z", "operational_available_at": "2026-08-22T12:00:03Z"}}]}
    return contract, evidence


def runtime(contract, evidence):
    return {"match_id": "constructed-future", "competition_id": "fixture-league", "category": "men_senior", "stage_type": "league",
            "entity_namespace": "fixture-provider", "kickoff": "2026-09-06T12:00:00Z", "prediction_cutoff": "2026-09-05T12:00:00Z",
            "fixture_available_at": "2026-09-04T12:00:00Z", "information_mode": "operational", "receipts": [],
            "contract_sha256": validate_prospective_contract(contract, evidence=evidence)["contract_sha256"]}


def codes(result):
    return {r["code"] for r in result["reasons"]}


def test_qualified_metadata_readiness_is_deterministic_readonly_and_never_execution():
    contract, evidence = fixture()
    before = deepcopy((contract, evidence))
    result = validate_prospective_contract(contract, evidence=evidence)
    assert result["status"] == "metadata_contract_ready" and not result["reasons"]
    assert result == validate_prospective_contract(contract, evidence=evidence)
    assert (contract, evidence) == before
    assert result["experiment_executed"] is False and result["external_requests"] == result["new_independent_matches"] == 0
    assert "authenticity" in result["scope"]


def test_historical_reconstruction_keeps_retrospective_status_and_cannot_be_relabelled():
    contract, evidence = fixture()
    contract["mode"] = "retrospective_exploratory"
    contract.pop("deployment")
    contract.pop("selection")
    contract.pop("freeze_at")
    contract.pop("domain")
    contract["information_semantics"]["information_mode"] = "historical_reconstruction"
    evidence["sources"] = []
    for art in evidence["artifacts"].values():
        art["information_mode"] = "historical_reconstruction"
    evidence["lambda_selection"]["information_mode"] = "historical_reconstruction"
    action = evidence["artifacts"]["catalog"]["actions"][0]
    action["cost_scope"] = "simulated_reveal"
    for key in action["expected_cost"]:
        action["expected_cost"][key]["basis"] = "simulated"
    for key, unit in (("money", "simulation_credit"), ("requests", "projection_call")):
        action["expected_cost"][key]["unit"] = contract["cost_units"][key] = unit
    result = validate_prospective_contract(contract, evidence=evidence)
    assert result["status"] == "retrospective_only"
    assert {"HISTORICAL_INFORMATION_MODE", "SIMULATED_OR_CACHED_PROJECTION", "NO_OPERATIONAL_ARCHIVE_SOURCE",
            "SIMULATED_COST_UNITS", "DEPLOYMENT_WINDOW_UNDECLARED", "SELECTION_RULE_UNDECLARED"} <= codes(result)
    contract["mode"] = "prospective_operational"
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "blocked"


@pytest.mark.parametrize("name,key", [("predictor", "max_data_available_at"), ("predictor", "available_at"),
    ("calibrator", "trained_as_of"), ("calibrator", "diagnostics_data_available_at"), ("catalog", "available_at")])
def test_all_model_and_calibration_availability_boundaries_precede_freeze(name, key):
    contract, evidence = fixture()
    evidence["artifacts"][name][key] = "2026-09-01T00:00:01Z"
    result = validate_prospective_contract(contract, evidence=evidence)
    assert result["status"] == "blocked" and "ARTIFACT_OR_DATA_AFTER_FREEZE" in codes(result)


@pytest.mark.parametrize("key", ["selected_at", "available_at", "max_data_available_at"])
def test_lambda_selection_input_availability_must_precede_freeze(key):
    contract, evidence = fixture()
    evidence["lambda_selection"][key] = "2026-09-03T00:00:00Z"
    assert "LAMBDA_AFTER_FREEZE" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_timezone_is_explicit_and_deployment_strictly_follows_freeze():
    contract, evidence = fixture()
    contract["deployment"]["start"] = "2026-09-01T08:00:00+08:00"
    assert "DEPLOYMENT_NOT_AFTER_FREEZE" in codes(validate_prospective_contract(contract, evidence=evidence))
    contract["deployment"]["start"] = "2026-09-02T00:00:00"
    assert "MISSING_OR_INVALID_TIME" in codes(validate_prospective_contract(contract, evidence=evidence))


@pytest.mark.parametrize("key", ["reference_predictor_sha256", "reference_predictor_version"])
def test_calibrator_must_bind_actual_reference_joint(key):
    contract, evidence = fixture()
    evidence["artifacts"]["calibrator"][key] = digest("other-model")
    assert "CALIBRATOR_PREDICTOR_BINDING_MISMATCH" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_hash_equality_does_not_hide_return_schema_or_cost_unit_changes():
    contract, evidence = fixture()
    evidence["artifacts"]["catalog"]["return_schema_sha256"] = digest("changed-tool")
    evidence["artifacts"]["catalog"]["actions"][0]["expected_cost"]["money"]["unit"] = "CNY"
    result = validate_prospective_contract(contract, evidence=evidence)
    assert {"TOOL_RETURN_SCHEMA_MISMATCH", "TOOL_COST_UNIT_MISMATCH"} <= codes(result)


def test_relabelling_value_as_multistep_or_replacing_artifact_is_blocked():
    contract, evidence = fixture()
    contract["policy"]["horizon"] = 2
    contract["artifacts"]["catalog"]["sha256"] = digest("substituted-catalog")
    result = validate_prospective_contract(contract, evidence=evidence)
    assert {"POLICY_VALUE_SCOPE_MISMATCH", "ARTIFACT_BINDING_MISMATCH"} <= codes(result)


def test_unknown_provider_price_stays_unknown_and_blocks_a_priced_utility_claim():
    contract, evidence = fixture()
    evidence["artifacts"]["catalog"]["actions"][0]["expected_cost"]["money"].update(amount=None, basis="unknown")
    result = validate_prospective_contract(contract, evidence=evidence)
    assert result["status"] == "blocked" and "COST_NOT_OPERATIONAL" in codes(result)
    assert evidence["artifacts"]["catalog"]["actions"][0]["expected_cost"]["money"]["amount"] is None


@pytest.mark.parametrize("where", ["predictor", "calibrator", "lambda", "source"])
def test_cross_domain_youth_cup_is_not_promoted_to_trained_senior_league(where):
    contract, evidence = fixture()
    target = evidence["artifacts"][where]["training_domain"] if where in {"predictor", "calibrator"} else (
        evidence["lambda_selection"]["training_domain"] if where == "lambda" else evidence["sources"][0]["domain"])
    target.update(category="men_under23", stage_types=["group_stage"])
    result = validate_prospective_contract(contract, evidence=evidence)
    assert result["status"] == "blocked" and any("DOMAIN" in c for c in codes(result))


@pytest.mark.parametrize("key", ["lambda_cost", "reporting_utility_weights"])
def test_policy_lambda_and_reporting_utility_are_separate_frozen_bindings(key):
    contract, evidence = fixture()
    contract["policy"][key]["money"] += .1
    assert "POLICY_WEIGHT_MISMATCH" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_fixed_preregistered_lambda_has_no_hidden_validation_dependency():
    contract, evidence = fixture()
    contract["policy"]["lambda_source"]["kind"] = evidence["lambda_selection"]["kind"] = "fixed_preregistered"
    for key in ("max_data_available_at", "split_manifest_sha256", "training_domain", "information_mode"):
        evidence["lambda_selection"].pop(key)
    evidence["lambda_selection"]["selected_at"] = "2026-07-01T00:00:00Z"
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "metadata_contract_ready"
    evidence["lambda_selection"]["max_data_available_at"] = "2026-08-19T00:00:00Z"
    assert "FIXED_LAMBDA_HAS_DATA_DEPENDENCY" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_free_text_selection_and_post_freeze_id_picking_are_rejected():
    contract, evidence = fixture()
    contract["selection"] = "we will choose unbiased matches"
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "blocked"
    contract, evidence = fixture()
    contract["selection"].update(kind="frozen_fixture_ids", fixture_ids=["constructed-future"], frozen_at="2026-09-03T00:00:00Z")
    assert "SELECTION_AFTER_FREEZE" in codes(validate_prospective_contract(contract, evidence=evidence))


@pytest.mark.parametrize("value", ["synthetic_fixture", "historical_reconstruction", "capability_declaration_only"])
def test_archive_capability_claim_is_not_observed_operational_proof(value):
    contract, evidence = fixture()
    evidence["sources"][0]["archive_basis"] = value
    assert "NONOPERATIONAL_ARCHIVE_SOURCE" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_archive_availability_and_verifier_timing_stay_ordered():
    contract, evidence = fixture()
    evidence["sources"][0]["probe"]["operational_available_at"] = "2026-08-22T11:00:00Z"
    assert "INVALID_OPERATIONAL_TIME_ORDER" in codes(validate_prospective_contract(contract, evidence=evidence))
    contract, evidence = fixture()
    evidence["sources"][0]["verified_at"] = "2026-09-01T01:00:00Z"
    assert "SOURCE_PROOF_AFTER_FREEZE" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_bound_artifact_and_lambda_selection_cannot_predate_their_dependencies():
    contract, evidence = fixture()
    evidence["artifacts"]["catalog"]["available_at"] = "2026-08-16T00:00:00Z"
    assert "CALIBRATOR_PRECEDES_REFERENCE_ARTIFACT" in codes(validate_prospective_contract(contract, evidence=evidence))
    contract, evidence = fixture()
    evidence["lambda_selection"]["selected_at"] = "2026-08-14T00:00:00Z"
    assert "LAMBDA_PRECEDES_REFERENCE_ARTIFACT" in codes(validate_prospective_contract(contract, evidence=evidence))


@pytest.mark.parametrize("section,field", [("contract", "mode"), ("policy", "risk_level"), ("lambda", "kind"), ("action", "source_id")])
def test_malformed_enum_and_identity_metadata_returns_blocked_instead_of_raising(section, field):
    contract, evidence = fixture()
    target = contract if section == "contract" else contract["policy"] if section == "policy" else (
        evidence["lambda_selection"] if section == "lambda" else evidence["artifacts"]["catalog"]["actions"][0])
    target[field] = {"invalid": "metadata"}
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "blocked"


@pytest.mark.parametrize("key", ["outcomes", "hidden_returns", "home_goals", "response_body", "joint_return_hypothesis"])
def test_runtime_rejects_outcomes_or_returns_at_any_depth_without_echoing_or_hashing_them(key):
    contract, evidence = fixture()
    meta = runtime(contract, evidence)
    meta["receipts"] = [{"times": {key: "PRIVATE_SENTINEL"}}]
    result = validate_runtime_metadata(contract, meta, evidence=evidence)
    assert result["status"] == "blocked" and "NONPUBLIC_PAYLOAD" in codes(result)
    assert result["runtime_metadata_sha256"] is None
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


def test_runtime_checks_only_public_fixture_and_archive_metadata_without_mutation():
    contract, evidence = fixture()
    meta = runtime(contract, evidence)
    before = deepcopy(meta)
    assert validate_runtime_metadata(contract, meta, evidence=evidence)["status"] == "runtime_metadata_admissible"
    assert meta == before
    meta["receipts"] = [{"evidence_id": "e1", "source_id": "constructed-source", "action_id": "results_view",
                         "times": deepcopy(evidence["sources"][0]["probe"])}]
    assert validate_runtime_metadata(contract, meta, evidence=evidence)["status"] == "runtime_metadata_admissible"
    meta["receipts"][0]["times"]["operational_available_at"] = "2026-09-05T12:00:01Z"
    assert "RUNTIME_EVIDENCE_AFTER_CUTOFF" in codes(validate_runtime_metadata(contract, meta, evidence=evidence))


@pytest.mark.parametrize("key,value,expected", [("category", "women_senior", "RUNTIME_OUT_OF_DOMAIN"),
    ("prediction_cutoff", "2026-09-06T12:00:00Z", "RUNTIME_OUTSIDE_PREDICTION_WINDOW"),
    ("fixture_available_at", "2026-09-05T12:00:01Z", "RUNTIME_OUTSIDE_PREDICTION_WINDOW"),
    ("information_mode", "historical_reconstruction", "RUNTIME_HISTORICAL_MODE"),
    ("contract_sha256", digest("other-policy"), "RUNTIME_POLICY_BINDING_MISMATCH")])
def test_runtime_window_identity_and_frozen_contract_are_enforced(key, value, expected):
    contract, evidence = fixture()
    meta = runtime(contract, evidence)
    meta[key] = value
    assert expected in codes(validate_runtime_metadata(contract, meta, evidence=evidence))


def test_missing_supporting_metadata_and_malformed_contract_are_blocked():
    assert validate_prospective_contract({}, evidence={})["status"] == "blocked"
    contract, evidence = fixture()
    evidence["sources"] = []
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "blocked"
    contract, evidence = fixture()
    contract["policy"]["lambda_cost"]["money"] = float("nan")
    assert "INVALID_METADATA_NUMBER" in codes(validate_prospective_contract(contract, evidence=evidence))


def test_retrospective_contract_does_not_enter_operational_runtime_even_with_complete_metadata():
    contract, evidence = fixture()
    meta = runtime(contract, evidence)
    contract["mode"] = "retrospective_exploratory"
    result = validate_runtime_metadata(contract, meta, evidence=evidence)
    assert result["status"] == "blocked" and "RETROSPECTIVE_RUNTIME_ONLY" in codes(result)


def test_validation_has_no_filesystem_or_network_side_effects(monkeypatch):
    import builtins
    import socket
    contract, evidence = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata validation must not perform I/O")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert validate_prospective_contract(contract, evidence=evidence)["status"] == "metadata_contract_ready"
