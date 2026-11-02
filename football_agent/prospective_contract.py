"""Offline, metadata-only eligibility checks for a frozen prospective policy.

Readiness qualifies the supplied metadata contract. Hashes bind content; they do
not establish source authenticity, launch a policy, or prove experiment success.
Caller-owned archive/artifact verifiers must produce the public evidence first.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re


VERSION = "prospective-policy-contract-v1"
VALUE_TARGET = "empirical_fixed_predictor_one_step_log_loss_reduction"
UNITS = ("money", "requests", "latency_ms", "tokens")
DOMAIN_KEYS = {"sport", "category", "competition_ids", "stage_types", "entity_namespace"}
FORBIDDEN = {"outcome", "outcomes", "label", "labels", "target_label", "home_goals", "away_goals", "result",
             "hidden_returns", "hidden_groups", "hidden_features", "full_snapshot", "raw_data", "payload", "groups",
             "joint_return_hypothesis", "probabilities", "prediction", "response_body", "body"}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class _Checks:
    def __init__(self):
        self.reasons = []

    def fail(self, code, path, message, *, gap=False):
        self.reasons.append({"code": code, "path": path, "message": message,
                             "kind": "prospective_gap" if gap else "invalid"})

    def obj(self, value, path, allowed, required=()):
        if not isinstance(value, dict):
            self.fail("MISSING_OR_INVALID_METADATA", path, "A public metadata object is required")
            return {}
        for key in sorted(set(value) - set(allowed)):
            self.fail("UNEXPECTED_METADATA_FIELD", path + "/" + key, "Field is outside the metadata allowlist")
        for key in sorted(set(required) - set(value)):
            self.fail("MISSING_METADATA_FIELD", path + "/" + key, "Required public metadata is missing")
        return value

    def time(self, value, path, *, gap=False):
        try:
            if not isinstance(value, str):
                raise ValueError
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is None or result.utcoffset() is None:
                raise ValueError
            return result.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            self.fail("MISSING_OR_INVALID_TIME", path, "An explicit timezone-aware timestamp is required", gap=gap)
            return None

    def hash(self, value, path):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            self.fail("INVALID_SHA256", path, "A lowercase SHA256 digest is required")
            return False
        return True

    def number(self, value, path):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            self.fail("INVALID_NONNEGATIVE_NUMBER", path, "A finite nonnegative numeric value is required")
            return False
        return True

    def names(self, value, path, *, nonempty=True):
        if not isinstance(value, list) or (nonempty and not value) or any(not isinstance(x, str) or not x.strip() for x in value):
            self.fail("INVALID_ID_LIST", path, "A list of explicit nonempty IDs is required")
            return []
        if len(value) != len(set(value)):
            self.fail("DUPLICATE_ID", path, "IDs must be unique")
        return value

    def public(self, value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    self.fail("INVALID_METADATA_KEY", path, "Metadata keys must be strings")
                    continue
                if key.casefold() in FORBIDDEN:
                    self.fail("NONPUBLIC_PAYLOAD", path + "/" + key, "Outcome or return content is outside the metadata boundary")
                self.public(item, path + "/" + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self.public(item, path + f"/{i}")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            self.fail("INVALID_METADATA_TYPE", path, "Only JSON metadata is accepted")
        elif isinstance(value, float) and not math.isfinite(value):
            self.fail("INVALID_METADATA_NUMBER", path, "Nonfinite metadata is rejected")

    def domain(self, value, path):
        if value is None:
            self.fail("TRAINING_OR_DEPLOYMENT_DOMAIN_UNDECLARED", path, "A structured domain declaration is required", gap=True)
            return None
        item = self.obj(value, path, DOMAIN_KEYS, DOMAIN_KEYS)
        for key in ("sport", "category", "entity_namespace"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                self.fail("INVALID_DOMAIN", path + "/" + key, "Explicit domain membership is required")
        result = {k: item.get(k) for k in ("sport", "category", "entity_namespace")}
        for key in ("competition_ids", "stage_types"):
            result[key] = sorted(self.names(item.get(key), path + "/" + key))
        return result

    def weights(self, value, path):
        item = self.obj(value, path, UNITS, UNITS)
        for key in UNITS:
            self.number(item.get(key), path + "/" + key)
        return item


def _report(checks, contract, evidence, *, runtime=False, metadata=None):
    mode = contract.get("mode") if isinstance(contract, dict) else None
    invalid = any(r["kind"] == "invalid" for r in checks.reasons)
    status = "blocked" if invalid or (mode == "prospective_operational" and checks.reasons) else (
        "retrospective_only" if mode == "retrospective_exploratory" else "metadata_contract_ready")
    if runtime and status == "metadata_contract_ready":
        status = "runtime_metadata_admissible"
    public = not any(r["code"] in {"NONPUBLIC_PAYLOAD", "INVALID_METADATA_KEY", "INVALID_METADATA_TYPE", "INVALID_METADATA_NUMBER"} for r in checks.reasons)
    return {"schema": VERSION, "status": status, "reasons": checks.reasons,
            "contract_sha256": _digest(contract) if public else None,
            "evidence_metadata_sha256": _digest(evidence) if public else None,
            **({"runtime_metadata_sha256": _digest(metadata) if public else None} if runtime else {}),
            "scope": "Metadata consistency only; caller verifier provenance and source authenticity are not independently established.",
            "experiment_executed": False, "external_requests": 0, "new_independent_matches": 0}


def validate_prospective_contract(contract, *, evidence):
    """Check declarations against independently supplied public verifier metadata.

    `retrospective_only` may contain prospective gaps. Malformed bindings are
    blocked in either mode. No missing field is filled with an invented default.
    """
    c = _Checks()
    c.public(contract, "contract")
    c.public(evidence, "evidence")
    if c.reasons:
        return _report(c, contract, evidence)
    con = c.obj(contract, "contract", {"schema", "mode", "freeze_at", "domain", "deployment", "selection", "artifacts",
        "value_target", "cost_units", "policy", "tools", "information_semantics"}, {"schema", "mode", "artifacts", "value_target", "cost_units", "policy", "tools"})
    ev = c.obj(evidence, "evidence", {"verifier", "artifacts", "lambda_selection", "sources"}, {"verifier", "artifacts", "lambda_selection", "sources"})
    if con.get("schema") != VERSION or con.get("mode") not in ("retrospective_exploratory", "prospective_operational"):
        c.fail("INVALID_CONTRACT_MODE", "contract", "An explicit version and retrospective/prospective mode are required")
    verifier = c.obj(ev.get("verifier"), "evidence/verifier", {"name", "version", "report_sha256"}, {"name", "version", "report_sha256"})
    c.hash(verifier.get("report_sha256"), "evidence/verifier/report_sha256")
    if any(not isinstance(verifier.get(k), str) or not verifier[k] for k in ("name", "version")):
        c.fail("VERIFIER_PROVENANCE_MISSING", "evidence/verifier", "Name and version of the caller's metadata verifier are required")
    if con.get("value_target") != VALUE_TARGET:
        c.fail("VALUE_TARGET_MISMATCH", "contract/value_target", "This contract supports empirical fixed-predictor one-step log-loss update gain")
    domain = c.domain(con.get("domain"), "contract/domain")
    freeze = c.time(con.get("freeze_at"), "contract/freeze_at", gap=True)
    deployment = con.get("deployment")
    start = end = None
    if deployment is None:
        c.fail("DEPLOYMENT_WINDOW_UNDECLARED", "contract/deployment", "An actual fixed prediction window has not been supplied", gap=True)
    else:
        dep = c.obj(deployment, "contract/deployment", {"start", "end", "min_lead_hours", "max_lead_hours"}, {"start", "end", "min_lead_hours", "max_lead_hours"})
        start, end = (c.time(dep.get(k), "contract/deployment/" + k) for k in ("start", "end"))
        a, b = (c.number(dep.get(k), "contract/deployment/" + k) for k in ("min_lead_hours", "max_lead_hours"))
        if a and b and (dep["min_lead_hours"] <= 0 or dep["min_lead_hours"] > dep["max_lead_hours"]):
            c.fail("INVALID_PREDICTION_LEAD", "contract/deployment", "A positive prematch lead interval is required")
        if start and end and start >= end:
            c.fail("INVALID_DEPLOYMENT_WINDOW", "contract/deployment", "Prediction start must precede end")
        if freeze and start and start <= freeze:
            c.fail("DEPLOYMENT_NOT_AFTER_FREEZE", "contract/deployment/start", "Prediction window must strictly follow policy freeze")
    selection = con.get("selection")
    if selection is None:
        c.fail("SELECTION_RULE_UNDECLARED", "contract/selection", "A structured fixed selection rule is required", gap=True)
    else:
        s = c.obj(selection, "contract/selection", {"kind", "time_field", "frozen_at", "competition_ids", "fixture_ids"}, {"kind", "time_field", "frozen_at", "competition_ids"})
        selected = c.names(s.get("competition_ids"), "contract/selection/competition_ids")
        if domain is not None and sorted(selected) != domain["competition_ids"]:
            c.fail("SELECTION_DOMAIN_MISMATCH", "contract/selection", "Fixed selection competitions must equal the declared domain")
        if s.get("kind") not in ("all_eligible", "frozen_fixture_ids") or s.get("time_field") != "prediction_cutoff":
            c.fail("UNSUPPORTED_SELECTION_RULE", "contract/selection", "Use all eligible fixtures or frozen IDs within the fixed prediction window")
        if s.get("kind") == "frozen_fixture_ids":
            c.names(s.get("fixture_ids"), "contract/selection/fixture_ids")
        elif "fixture_ids" in s:
            c.fail("SELECTION_RULE_CONFLICT", "contract/selection/fixture_ids", "all_eligible has no handpicked ID list")
        at = c.time(s.get("frozen_at"), "contract/selection/frozen_at")
        if freeze and at and at > freeze:
            c.fail("SELECTION_AFTER_FREEZE", "contract/selection/frozen_at", "Selection rule was fixed after policy freeze")
    units = c.obj(con.get("cost_units"), "contract/cost_units", UNITS, UNITS)
    if any(not isinstance(units.get(u), str) or not units[u] for u in UNITS):
        c.fail("COST_UNITS_UNDECLARED", "contract/cost_units", "All physical cost units must be explicit")
    policy = c.obj(con.get("policy"), "contract/policy", {"horizon", "risk_level", "lambda_cost", "lambda_source", "reporting_utility_weights"},
                   {"horizon", "risk_level", "lambda_cost", "lambda_source", "reporting_utility_weights"})
    if type(policy.get("horizon")) is not int or policy["horizon"] != 1 or policy.get("risk_level") not in ("mean", "conservative"):
        c.fail("POLICY_VALUE_SCOPE_MISMATCH", "contract/policy", "The empirical gate is one step with mean or descriptive LCB")
    c.weights(policy.get("lambda_cost"), "contract/policy/lambda_cost")
    c.weights(policy.get("reporting_utility_weights"), "contract/policy/reporting_utility_weights")
    lambda_ref = c.obj(policy.get("lambda_source"), "contract/policy/lambda_source", {"kind", "sha256"}, {"kind", "sha256"})
    c.hash(lambda_ref.get("sha256"), "contract/policy/lambda_source/sha256")
    tools = c.names(con.get("tools"), "contract/tools")
    refs = c.obj(con.get("artifacts"), "contract/artifacts", {"predictor", "calibrator", "catalog"}, {"predictor", "calibrator", "catalog"})
    arts = c.obj(ev.get("artifacts"), "evidence/artifacts", {"predictor", "calibrator", "catalog"}, {"predictor", "calibrator", "catalog"})
    checked = {}
    common = {"sha256", "version", "available_at", "information_mode"}
    for name in ("predictor", "calibrator", "catalog"):
        path = "evidence/artifacts/" + name
        ref = c.obj(refs.get(name), "contract/artifacts/" + name, {"sha256", "version"}, {"sha256", "version"})
        extra = ({"trained_as_of", "max_data_available_at", "training_domain", "return_schema_sha256", "action_ids"} if name == "predictor" else
                 {"trained_as_of", "max_data_available_at", "diagnostics_data_available_at", "training_domain", "reference_predictor_sha256",
                  "reference_predictor_version", "reference_catalog_sha256", "value_target"} if name == "calibrator" else {"actions", "return_schema_sha256"})
        item = c.obj(arts.get(name), path, common | extra, common)
        checked[name] = item
        for owner, prefix in ((ref, "contract/artifacts/" + name), (item, path)):
            c.hash(owner.get("sha256"), prefix + "/sha256")
            if not isinstance(owner.get("version"), str) or not owner["version"]:
                c.fail("ARTIFACT_VERSION_MISSING", prefix + "/version", "Artifact version is required")
        if any(ref.get(k) != item.get(k) for k in ("sha256", "version")):
            c.fail("ARTIFACT_BINDING_MISMATCH", path, "Configured artifact and supplied verifier identity differ")
        times = ["available_at"] + (["trained_as_of", "max_data_available_at"] if name != "catalog" else [])
        if name == "calibrator":
            times.append("diagnostics_data_available_at")
        stamps = {key: c.time(item.get(key), path + "/" + key, gap=True) for key in times}
        for key, at in stamps.items():
            if freeze and at and at > freeze:
                c.fail("ARTIFACT_OR_DATA_AFTER_FREEZE", path + "/" + key, "Every artifact and data availability boundary must precede freeze")
        if stamps.get("max_data_available_at") and stamps.get("trained_as_of") and stamps["max_data_available_at"] > stamps["trained_as_of"]:
            c.fail("TRAINING_DATA_NOT_YET_AVAILABLE", path, "Fitting predates the availability of its inputs")
        if stamps.get("available_at") and any(at > stamps["available_at"] for key, at in stamps.items() if at is not None and key != "available_at"):
            c.fail("ARTIFACT_PRECEDES_DEPENDENCIES", path, "Artifact availability predates its declared dependencies")
        if item.get("information_mode") != "operational":
            c.fail("HISTORICAL_INFORMATION_MODE", path + "/information_mode", "Historical reconstruction is not operational acquisition evidence", gap=True)
        if name != "catalog":
            observed_domain = c.domain(item.get("training_domain"), path + "/training_domain")
            if domain is not None and observed_domain is not None and domain != observed_domain:
                c.fail("OUT_OF_TRAINING_DOMAIN", path + "/training_domain", "Deployment domain differs from the fitted domain", gap=True)
    pred, cal, catalog = (checked[k] for k in ("predictor", "calibrator", "catalog"))
    # Reference parameters may be fitted earlier, but their final bound artifact
    # cannot be published before its referenced model/catalog exist.
    published = {}
    for name, item in checked.items():
        raw = item.get("available_at")
        if isinstance(raw, str):
            try:
                at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if at.tzinfo is not None:
                    published[name] = at
            except ValueError:
                pass
    if "calibrator" in published and any(published[k] > published["calibrator"] for k in ("predictor", "catalog") if k in published):
        c.fail("CALIBRATOR_PRECEDES_REFERENCE_ARTIFACT", "evidence/artifacts/calibrator/available_at", "Bound calibrator artifact predates its model or catalog")
    for a, b in (("reference_predictor_sha256", "sha256"), ("reference_predictor_version", "version")):
        if cal.get(a) != pred.get(b) or cal.get(a) is None:
            c.fail("CALIBRATOR_PREDICTOR_BINDING_MISMATCH", "evidence/artifacts/calibrator/" + a, "Calibrator must reference the exact predictor")
    if cal.get("reference_catalog_sha256") != catalog.get("sha256") or cal.get("reference_catalog_sha256") is None:
        c.fail("CALIBRATOR_CATALOG_UNBOUND", "evidence/artifacts/calibrator/reference_catalog_sha256", "Calibrated actions must bind the deployed catalog", gap=True)
    if cal.get("value_target") != con.get("value_target"):
        c.fail("CALIBRATOR_TARGET_MISMATCH", "evidence/artifacts/calibrator/value_target", "Estimator semantics differ from the declared one-step target")
    if pred.get("return_schema_sha256") is None:
        c.fail("MODEL_RETURN_SCHEMA_UNBOUND", "evidence/artifacts/predictor/return_schema_sha256", "A fitted return-encoding schema binding is missing", gap=True)
    else:
        c.hash(pred.get("return_schema_sha256"), "evidence/artifacts/predictor/return_schema_sha256")
    if pred.get("return_schema_sha256") != catalog.get("return_schema_sha256"):
        c.fail("TOOL_RETURN_SCHEMA_MISMATCH", "evidence/artifacts/catalog", "Learned return encoding differs from the deployed tool schema", gap=True)
    if sorted(c.names(pred.get("action_ids"), "evidence/artifacts/predictor/action_ids")) != sorted(tools):
        c.fail("MODEL_ACTIONS_MISMATCH", "contract/tools", "Available actions differ from the modeled actions", gap=True)
    _check_lambda(c, ev.get("lambda_selection"), lambda_ref, policy, checked, domain, freeze)
    _check_sources(c, con, ev.get("sources"), catalog, tools, units, domain, freeze)
    return _report(c, con, ev)


def _check_lambda(c, value, ref, policy, arts, domain, freeze):
    path = "evidence/lambda_selection"
    keys = {"sha256", "kind", "available_at", "selected_at", "max_data_available_at", "information_mode", "training_domain",
            "predictor_sha256", "calibrator_sha256", "catalog_sha256", "lambda_cost", "reporting_utility_weights", "split_manifest_sha256"}
    item = c.obj(value, path, keys, keys - {"split_manifest_sha256", "max_data_available_at", "training_domain", "information_mode"})
    c.hash(item.get("sha256"), path + "/sha256")
    if item.get("sha256") != ref.get("sha256") or item.get("kind") != ref.get("kind"):
        c.fail("LAMBDA_SOURCE_BINDING_MISMATCH", path, "Policy lambda source differs from the frozen selection record")
    if item.get("kind") not in ("heldout_temporal_validation", "fixed_preregistered"):
        c.fail("INVALID_LAMBDA_SOURCE", path + "/kind", "Lambda needs temporal validation or a fixed preregistration record")
    for name in ("predictor", "calibrator", "catalog"):
        if item.get(name + "_sha256") != arts[name].get("sha256"):
            c.fail("LAMBDA_ARTIFACT_MISMATCH", path + "/" + name + "_sha256", "Lambda was selected for a different artifact")
    for key in ("lambda_cost", "reporting_utility_weights"):
        c.weights(item.get(key), path + "/" + key)
        if item.get(key) != policy.get(key):
            c.fail("POLICY_WEIGHT_MISMATCH", path + "/" + key, "Frozen selection/reporting weights differ from policy configuration")
    stamps = {key: c.time(item.get(key), path + "/" + key, gap=True) for key in ("available_at", "selected_at")}
    if item.get("kind") == "heldout_temporal_validation":
        c.hash(item.get("split_manifest_sha256"), path + "/split_manifest_sha256")
        stamps["max_data_available_at"] = c.time(item.get("max_data_available_at"), path + "/max_data_available_at")
        selected_domain = c.domain(item.get("training_domain"), path + "/training_domain")
        if domain is not None and selected_domain is not None and selected_domain != domain:
            c.fail("LAMBDA_DOMAIN_MISMATCH", path, "Validation and deployment domains differ", gap=True)
        if item.get("information_mode") != "operational":
            c.fail("LAMBDA_HISTORICAL_INFORMATION_MODE", path, "Lambda validation used historical reconstruction", gap=True)
        if stamps["max_data_available_at"] and stamps["selected_at"] and stamps["max_data_available_at"] > stamps["selected_at"]:
            c.fail("LAMBDA_DATA_NOT_YET_AVAILABLE", path, "Selection predates validation-data availability")
    elif item.get("max_data_available_at") is not None or item.get("split_manifest_sha256") is not None:
        c.fail("FIXED_LAMBDA_HAS_DATA_DEPENDENCY", path, "Data-selected weights are not fixed preregistration")
    for key, at in stamps.items():
        if freeze and at and at > freeze:
            c.fail("LAMBDA_AFTER_FREEZE", path + "/" + key, "Lambda selection and every input availability must precede freeze")
    reference_time = stamps.get("selected_at") if item.get("kind") == "heldout_temporal_validation" else stamps.get("available_at")
    if reference_time:
        for name, art in arts.items():
            raw = art.get("available_at")
            if not isinstance(raw, str):
                continue
            try:
                at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if at.tzinfo is not None and at > reference_time:
                    c.fail("LAMBDA_PRECEDES_REFERENCE_ARTIFACT", path, "Selection or final preregistered binding predates a referenced artifact")
            except ValueError:
                pass
    if stamps.get("available_at") and stamps.get("selected_at") and stamps["available_at"] < stamps["selected_at"]:
        c.fail("LAMBDA_RECORD_PRECEDES_SELECTION", path, "Selection record predates the selection")


def _check_sources(c, con, sources, catalog, tools, units, domain, freeze):
    semantic = con.get("information_semantics")
    required = {"information_mode": "operational", "available_at_rule": "max_observed_ingested_registered",
                "pre_action_decision_archive": True, "separate_outcome_store": True}
    if semantic is None:
        c.fail("OPERATIONAL_TIME_SEMANTICS_MISSING", "contract/information_semantics", "Actual acquisition and registration timing is required", gap=True)
    else:
        item = c.obj(semantic, "contract/information_semantics", required, required)
        if item != required or item.get("pre_action_decision_archive") is not True or item.get("separate_outcome_store") is not True:
            c.fail("NONOPERATIONAL_TIME_SEMANTICS", "contract/information_semantics", "Assumed historical availability is not operational availability", gap=True)
    source_map = {}
    if not isinstance(sources, list) or not sources:
        c.fail("NO_OPERATIONAL_ARCHIVE_SOURCE", "evidence/sources", "Verified public metadata for an operational archive source is missing", gap=True)
        sources = []
    for i, source in enumerate(sources):
        path = f"evidence/sources/{i}"
        keys = {"source_id", "domain", "information_mode", "archive_basis", "archive_manifest_sha256", "verified_at", "probe",
                "records_request_response_costs", "records_pre_action_intent"}
        s = c.obj(source, path, keys, keys)
        ident = s.get("source_id")
        if not isinstance(ident, str) or not ident or ident in source_map:
            c.fail("INVALID_SOURCE_ID", path, "Unique source IDs are required")
        else:
            source_map[ident] = s
        c.hash(s.get("archive_manifest_sha256"), path + "/archive_manifest_sha256")
        at = c.time(s.get("verified_at"), path + "/verified_at")
        if freeze and at and at > freeze:
            c.fail("SOURCE_PROOF_AFTER_FREEZE", path, "Operational capability evidence postdates policy freeze")
        source_domain = c.domain(s.get("domain"), path + "/domain")
        if source_domain is not None and domain is not None and source_domain != domain:
            c.fail("SOURCE_DOMAIN_MISMATCH", path, "Source archive domain differs from fitted/deployment domain", gap=True)
        if s.get("information_mode") != "operational" or s.get("archive_basis") != "observed_operational_archive":
            c.fail("NONOPERATIONAL_ARCHIVE_SOURCE", path, "Synthetic fixtures, declarations and reconstructions are not operational archive evidence", gap=True)
        if s.get("records_request_response_costs") is not True or s.get("records_pre_action_intent") is not True:
            c.fail("OPERATIONAL_AUDIT_CAPABILITY_MISSING", path, "Request/return/cost and pre-action intent receipts are required", gap=True)
        probe = _check_receipt(c, s.get("probe"), path + "/probe")
        if at and probe and probe > at:
            c.fail("SOURCE_PROOF_PRECEDES_ARCHIVE", path, "Verifier timestamp precedes archive availability")
    actions = catalog.get("actions")
    if not isinstance(actions, list):
        c.fail("TOOL_CATALOG_MISSING", "evidence/artifacts/catalog/actions", "Explicit tool metadata is required")
        actions = []
    found = []
    for i, action in enumerate(actions):
        path = f"evidence/artifacts/catalog/actions/{i}"
        a = c.obj(action, path, {"action_id", "source_id", "cost_scope", "expected_cost"}, {"action_id", "source_id", "cost_scope", "expected_cost"})
        found.append(a.get("action_id"))
        if not isinstance(a.get("source_id"), str) or a["source_id"] not in source_map:
            c.fail("TOOL_OPERATIONAL_SOURCE_UNBOUND", path + "/source_id", "Tool has no matching operational archive source metadata", gap=True)
        if a.get("cost_scope") != "real_external_request":
            c.fail("SIMULATED_OR_CACHED_PROJECTION", path + "/cost_scope", "A cached view reveal is not an actual external acquisition", gap=True)
        costs = c.obj(a.get("expected_cost"), path + "/expected_cost", UNITS, UNITS)
        for unit in UNITS:
            v = c.obj(costs.get(unit), path + "/expected_cost/" + unit, {"amount", "unit", "basis"}, {"amount", "unit", "basis"})
            if v.get("unit") != units.get(unit):
                c.fail("TOOL_COST_UNIT_MISMATCH", path + "/expected_cost/" + unit, "Catalog cost units differ from policy units")
            if v.get("basis") not in ("known", "estimated"):
                c.fail("COST_NOT_OPERATIONAL", path + "/expected_cost/" + unit, "Simulated or unknown costs do not establish this operational utility contract", gap=True)
            if v.get("amount") is None and v.get("basis") == "unknown":
                continue
            c.number(v.get("amount"), path + "/expected_cost/" + unit + "/amount")
    c.names(found, "evidence/artifacts/catalog/action_ids")
    if sorted(x for x in found if isinstance(x, str)) != sorted(tools):
        c.fail("TOOL_CATALOG_SELECTION_MISMATCH", "contract/tools", "Contract actions differ from the bound catalog")
    if units.get("requests") != "HTTP_request" or units.get("money") == "simulation_credit":
        c.fail("SIMULATED_COST_UNITS", "contract/cost_units", "Projection calls/simulation credits are not external requests/provider currency", gap=True)


def _check_receipt(c, value, path):
    keys = {"sha256", "observed_at", "ingested_at", "registered_at", "operational_available_at"}
    item = c.obj(value, path, keys, keys)
    c.hash(item.get("sha256"), path + "/sha256")
    stamps = {key: c.time(item.get(key), path + "/" + key) for key in keys - {"sha256"}}
    if all(stamps.values()):
        if not stamps["observed_at"] <= stamps["ingested_at"] <= stamps["registered_at"] <= stamps["operational_available_at"]:
            c.fail("INVALID_OPERATIONAL_TIME_ORDER", path, "Observation, ingestion, registration and usable time must remain ordered")
    return stamps.get("operational_available_at")


def validate_runtime_metadata(contract, metadata, *, evidence):
    """Validate only fixture identity and receipt timestamps, never tool returns."""
    base = validate_prospective_contract(contract, evidence=evidence)
    c = _Checks()
    c.reasons.extend(base["reasons"])
    c.public(metadata, "runtime")
    if c.reasons or base["status"] != "metadata_contract_ready":
        if base["status"] == "retrospective_only":
            c.fail("RETROSPECTIVE_RUNTIME_ONLY", "contract/mode", "This contract is not a prospective operational policy")
        return _report(c, contract, evidence, runtime=True, metadata=metadata)
    keys = {"match_id", "competition_id", "category", "stage_type", "entity_namespace", "kickoff", "prediction_cutoff",
            "fixture_available_at", "information_mode", "receipts", "contract_sha256"}
    item = c.obj(metadata, "runtime", keys, keys)
    if item.get("contract_sha256") != base["contract_sha256"]:
        c.fail("RUNTIME_POLICY_BINDING_MISMATCH", "runtime/contract_sha256", "Runtime metadata belongs to another policy contract")
    cutoff, kickoff, fixture_at = (c.time(item.get(k), "runtime/" + k) for k in ("prediction_cutoff", "kickoff", "fixture_available_at"))
    dep, domain, selection = contract["deployment"], contract["domain"], contract["selection"]
    start, end = (datetime.fromisoformat(dep[k].replace("Z", "+00:00")) for k in ("start", "end"))
    if cutoff and kickoff and fixture_at:
        lead = (kickoff - cutoff).total_seconds()/3600
        if not start <= cutoff < end or not dep["min_lead_hours"] <= lead <= dep["max_lead_hours"] or fixture_at > cutoff:
            c.fail("RUNTIME_OUTSIDE_PREDICTION_WINDOW", "runtime", "Fixture/time metadata is outside the frozen prematch window")
    if (item.get("competition_id") not in domain["competition_ids"] or item.get("category") != domain["category"]
            or item.get("stage_type") not in domain["stage_types"] or item.get("entity_namespace") != domain["entity_namespace"]):
        c.fail("RUNTIME_OUT_OF_DOMAIN", "runtime", "Fixture domain differs from the frozen policy")
    if not isinstance(item.get("match_id"), str) or not item["match_id"] or (selection["kind"] == "frozen_fixture_ids" and item["match_id"] not in selection["fixture_ids"]):
        c.fail("RUNTIME_SELECTION_MISMATCH", "runtime/match_id", "Fixture is outside the fixed selection rule")
    if item.get("information_mode") != "operational":
        c.fail("RUNTIME_HISTORICAL_MODE", "runtime/information_mode", "Historical reconstruction is outside operational execution")
    receipts = item.get("receipts")
    if not isinstance(receipts, list):
        c.fail("RUNTIME_RECEIPTS_INVALID", "runtime/receipts", "A metadata receipt list is required")
        receipts = []
    action_sources = {a["action_id"]: a["source_id"] for a in evidence["artifacts"]["catalog"]["actions"]}
    seen = set()
    for i, value in enumerate(receipts):
        path = f"runtime/receipts/{i}"
        r = c.obj(value, path, {"evidence_id", "source_id", "action_id", "times"}, {"evidence_id", "source_id", "action_id", "times"})
        if not isinstance(r.get("evidence_id"), str) or not r["evidence_id"] or r["evidence_id"] in seen:
            c.fail("RUNTIME_DUPLICATE_OR_INVALID_EVIDENCE", path, "Receipt identity must be unique")
        else:
            seen.add(r["evidence_id"])
        if not isinstance(r.get("action_id"), str) or action_sources.get(r["action_id"]) != r.get("source_id"):
            c.fail("RUNTIME_TOOL_SOURCE_MISMATCH", path, "Receipt source/action is not in the frozen catalog")
        available = _check_receipt(c, r.get("times"), path + "/times")
        if cutoff and available and available > cutoff:
            c.fail("RUNTIME_EVIDENCE_AFTER_CUTOFF", path, "Evidence was unavailable at the prediction cutoff")
    return _report(c, contract, evidence, runtime=True, metadata=metadata)
