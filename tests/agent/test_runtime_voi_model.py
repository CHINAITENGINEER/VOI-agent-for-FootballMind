"""History package boundaries for the configured, state-bound runtime selector."""
from copy import deepcopy
import json

import pytest

from football_agent.runtime_voi_model import RuntimeVoIModel, VERSION
from football_agent.state_bound_context_value import NO_MATCH
from football_agent.voi_evidence_bridge import VoIEvidenceLink


def package():
    link = VoIEvidenceLink(
        link_id="link-1", trace_id="trace-1", episode_id="episode-1", target_key="spftll:17",
        request_id="request-1", action_id="discover-fixture", candidate_id="fixture-route",
        evidence_entry_id="entry-1", evidence_availability="available", evidence_quality="VALIDATED",
        requested_at="2026-09-20T09:00:00Z", source_version="v1", source_id="spftll-recent",
        prediction_cutoff="2026-09-20T10:00:00Z", expected_information_value=None,
        observed_risk_reduction=0.3, calibration_error=None, status="linked_available",
        dependency_key="source-event-1")
    return {"version": VERSION, "evidence_basis": "synthetic",
            "calibration_as_of": "2026-09-29T00:00:00Z", "loss_unit": "decision_loss",
            "context_keys": ["operation_document_count"], "min_support": 5,
            "confidence": 0.9, "family_size": 1,
            "routes": [{"route_id": "fixture-route", "match": {"action": "read_url", "url": "https://example.test/fixtures"},
                        "source_id": "spftll-recent", "source_version": "v1", "action_id": "discover-fixture",
                        "cost": 0.1, "unit": "decision_loss"}],
            "history": [{"link": link.as_dict(), "context": {"operation_document_count": 0},
                         "label_available_at": "2026-09-21T00:00:00Z"}]}


def test_package_exposes_explicit_provenance_without_claiming_calibration():
    model = RuntimeVoIModel(package())
    descriptor = model.configuration()
    assert model.evidence_basis == "synthetic"
    assert model.loss_unit == "decision_loss"
    assert model.context_keys == ("operation_document_count",)
    assert model.calibration_as_of == "2026-09-29T00:00:00+00:00"
    assert descriptor["numeric_voi_calibrated"] is False
    assert descriptor["evidence_basis_verified"] is False
    assert descriptor["synthetic_live_use_permitted"] is False
    assert descriptor["history_count"] == 1
    assert model.estimator.routes["fixture-route"]["cost"] == 0.1


def test_independently_evaluated_is_only_a_provenance_declaration():
    config = package()
    config["evidence_basis"] = "independently_evaluated"
    descriptor = RuntimeVoIModel(config).configuration()
    assert descriptor["evidence_basis"] == "independently_evaluated"
    assert descriptor["evidence_basis_verified"] is False
    assert descriptor["numeric_voi_calibrated"] is False


def test_input_routes_descriptors_and_estimator_copies_do_not_mutate_model():
    config = package()
    before = deepcopy(config)
    model = RuntimeVoIModel(config)
    descriptor = model.configuration()
    config["routes"][0]["cost"] = 99
    model.routes[0]["cost"] = 88
    copy = model.configuration()
    copy["routes"][0]["cost"] = 77
    estimator = model.estimator
    estimator.routes["fixture-route"]["cost"] = 66
    estimator.context_by_link_id["link-1"]["operation_document_count"] = 50
    assert model.configuration() == descriptor
    assert model.estimator.routes["fixture-route"]["cost"] == before["routes"][0]["cost"]
    assert model.estimator.context_by_link_id["link-1"]["operation_document_count"] == 0


def test_family_size_changes_package_identity_even_when_core_snapshot_omits_it():
    first = package()
    second = deepcopy(first)
    second["family_size"] = 3
    assert RuntimeVoIModel(first).configuration()["model_hash"] != RuntimeVoIModel(second).configuration()["model_hash"]


def test_labels_and_context_also_change_package_identity():
    first = package()
    second = deepcopy(first)
    second["history"][0]["label_available_at"] = "2026-09-22T00:00:00Z"
    assert RuntimeVoIModel(first).configuration()["model_hash"] != RuntimeVoIModel(second).configuration()["model_hash"]
    second = deepcopy(first)
    second["history"][0]["context"]["operation_document_count"] = 1
    assert RuntimeVoIModel(first).configuration()["model_hash"] != RuntimeVoIModel(second).configuration()["model_hash"]


def test_empty_history_never_supplies_implicit_synthetic_samples():
    config = package()
    config["history"] = []
    model = RuntimeVoIModel(config)
    interval = model.estimator.estimate_candidate("fixture-route", {"operation_document_count": 0},
                                               cutoff="2026-10-01T00:00:00Z", target_key="spftll:18")
    assert interval.support_status == NO_MATCH
    assert interval.mean_value is None


def test_current_canonical_target_is_excluded_by_estimator():
    interval = RuntimeVoIModel(package()).estimator.estimate_candidate(
        "fixture-route", {"operation_document_count": 0}, cutoff="2026-10-01T00:00:00Z", target_key="spftll:17")
    assert interval.support_status == NO_MATCH


def test_signed_risk_reduction_is_preserved():
    config = package()
    config["history"][0]["link"]["observed_risk_reduction"] = -0.8
    assert RuntimeVoIModel(config).estimator.links[0].observed_risk_reduction == -0.8


def test_search_query_template_supports_only_explicit_match_input():
    config = package()
    config["routes"][0]["match"] = {"action": "search_web", "engine": "news", "query_template": "{match_input} lineup"}
    assert RuntimeVoIModel(config).routes[0]["match"]["query_template"] == "{match_input} lineup"


@pytest.mark.parametrize("template", ["{query}", "{match_input.x}", "{match_input[0]}", "{match_input!r}", "{match_input:>8}", "{broken", "{{match_input}}", "{{escaped}}"])
def test_reject_undeclared_or_executable_template_fields(template):
    config = package()
    config["routes"][0]["match"] = {"action": "search_web", "engine": "bing", "query_template": template}
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("field,value", [
    ("version", "other"), ("evidence_basis", "live"), ("loss_unit", "http_calls"),
    ("loss_unit", "log_loss"), ("loss_unit", "brier"),
    ("evidence_basis", []), ("loss_unit", {}),
    ("min_support", True), ("min_support", 1.5), ("min_support", 0),
    ("family_size", True), ("family_size", 1.0), ("family_size", 0),
    ("confidence", True), ("confidence", 0), ("confidence", 1), ("confidence", float("nan")),
    ("calibration_as_of", "2026-09-29T00:00:00"),
    ("context_keys", ["outcome"]), ("context_keys", []),
    ("context_keys", ["operation_document_count", "operation_document_count"]),
])
def test_reject_invalid_configuration_parameters(field, value):
    config = package()
    config[field] = value
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("location", ["configuration", "route", "match", "history", "link", "context"])
def test_reject_unknown_fields_at_every_schema_boundary(location):
    config = package()
    target = {"configuration": config, "route": config["routes"][0], "match": config["routes"][0]["match"],
              "history": config["history"][0], "link": config["history"][0]["link"],
              "context": config["history"][0]["context"]}[location]
    target["evaluator"] = {"outcome": "home"}
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("field,value", [
    ("route_id", " fixture-route"), ("source_id", " source"), ("source_version", ""),
    ("cost", -0.1), ("cost", True), ("cost", float("inf")), ("unit", "http_calls"),
])
def test_reject_ambiguous_route_identity_or_cost(field, value):
    config = package()
    config["routes"][0][field] = value
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("field,value", [
    ("target_key", "spftll:017"), ("target_key", "spftll:0"), ("target_key", "teams:a-b"),
    ("target_key", "spftll:18 "), ("candidate_id", "unknown-route"),
    ("source_id", "other"), ("source_version", "v2"), ("action_id", "other"),
    ("observed_risk_reduction", 1.1), ("observed_risk_reduction", -1.1),
    ("observed_risk_reduction", True), ("observed_risk_reduction", None),
    ("hidden_result_included", True), ("hidden_result_included", 0),
    ("evidence_quality", "unverified"), ("evidence_entry_id", None),
    ("evidence_availability", "unavailable"), ("status", "stop_recorded"),
    ("requested_at", None), ("requested_at", "2026-09-22T00:00:00Z"),
    ("requested_at", "2026-09-20T12:00:00Z"),
    ("prediction_cutoff", "2026-09-22T00:00:00Z"),
])
def test_reject_misaligned_or_unusable_history(field, value):
    config = package()
    config["history"][0]["link"][field] = value
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("label", ["2026-09-19T00:00:00Z", "2026-09-30T00:00:00Z", "2026-09-21T00:00:00"])
def test_label_availability_must_be_timezone_bound_between_cutoff_and_calibration(label):
    config = package()
    config["history"][0]["label_available_at"] = label
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("gain", [None, 0, -0.2])
def test_unusable_attempts_allow_only_missing_or_zero_gain(gain):
    config = package()
    config["history"][0]["link"].update(status="linked_unusable", evidence_availability="unavailable",
                                         evidence_entry_id=None, evidence_quality=None, observed_risk_reduction=gain)
    if gain in (None, 0):
        assert RuntimeVoIModel(config).estimator.links[0].observed_risk_reduction == gain
    else:
        with pytest.raises(ValueError):
            RuntimeVoIModel(config)


def test_duplicate_link_and_route_identity_are_rejected():
    config = package()
    config["history"].append(deepcopy(config["history"][0]))
    with pytest.raises(ValueError, match="duplicate link_id"):
        RuntimeVoIModel(config)
    config = package()
    config["routes"].append(deepcopy(config["routes"][0]))
    with pytest.raises(ValueError, match="duplicate route_id"):
        RuntimeVoIModel(config)


def test_duplicate_exact_matcher_cannot_borrow_another_route_support():
    config = package()
    other = deepcopy(config["routes"][0])
    other["route_id"] = "another-route"
    config["routes"].append(other)
    with pytest.raises(ValueError, match="duplicate exact route matcher"):
        RuntimeVoIModel(config)


def test_family_size_covers_every_registered_route():
    config = package()
    other = deepcopy(config["routes"][0])
    other.update(route_id="another-route", match={"action": "read_url", "url": "https://example.test/other"})
    config["routes"].append(other)
    with pytest.raises(ValueError, match="family_size"):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("value", [True, -1, 0.5, "0", {"outcome": "home"}])
def test_operational_counts_are_typed_and_never_nested_labels(value):
    config = package()
    config["history"][0]["context"]["operation_document_count"] = value
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


def test_readable_count_does_not_exceed_all_documents():
    config = package()
    config["context_keys"].append("operation_readable_document_count")
    config["history"][0]["context"]["operation_readable_document_count"] = 1
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


def test_file_loader_keeps_identity_and_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "model.json"
    config = package()
    path.write_text(json.dumps(config), encoding="utf-8")
    assert RuntimeVoIModel.from_path(path).configuration() == RuntimeVoIModel(config).configuration()
    path.write_text('{"version":"x","version":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        RuntimeVoIModel.from_path(path)


def test_no_implicit_defaults_or_non_json_collections():
    config = package()
    del config["history"]
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)
    config = package()
    config["context_keys"] = tuple(config["context_keys"])
    with pytest.raises(ValueError):
        RuntimeVoIModel(config)


@pytest.mark.parametrize("source_id,source_version", [
    ("spftll", "spftll-agent-projection-v1"),
    ("search:bing", "experience-web-tools-v1"),
    ("search:news", "experience-web-tools-v1"),
    ("web:example.test", "experience-web-tools-v1"),
])
def test_explicit_executor_source_namespaces_are_accepted(source_id, source_version):
    config = package()
    config["routes"][0].update(source_id=source_id, source_version=source_version)
    config["history"][0]["link"].update(source_id=source_id, source_version=source_version)
    model = RuntimeVoIModel(config)
    assert model.estimator.routes["fixture-route"]["source_id"] == source_id
    assert model.estimator.routes["fixture-route"]["source_version"] == source_version
