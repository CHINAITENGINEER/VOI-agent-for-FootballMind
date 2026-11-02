"""Research g for raw-bound prospective frozen-f quality targets.

The existing stream outcome join owns label admission. This consumer reruns
that join, extracts features ONLY from each saved pre-I/O menu, and preserves
the pre-registered physical-match split and the complete prediction pipeline.
It reuses ContextValueModel's fixed ridge/TensorSketch kernel, not its older
input contract or online support gate. No acquisition or activation occurs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from football_research.data_adapters.predictor_context_pairs import _payload
from football_research.data_adapters.prospective_outcomes import derive_g_candidates
from .context_value import ContextValueModel, PARAMETERS, UNITS, _bounded
from .context_predictor import _digest
from .entity_semantic_f import FEATURES, REPRESENTATION_SHA256
from .state import parse_time

VERSION = "prospective-frozen-f-quality-g-v1"
QUERY_VERSION = "prospective-frozen-f-quality-query-v1"
DATA_VERSION = "prospective-frozen-f-quality-dataset-v1"
TARGET = "conditional_expected_gross_log_loss_reduction"
SCOPE = "entire_action_visible_state_change_including_exposed_links_and_body"
CONDITIONING = "both_f_evaluations_completed_on_time; extraction_failures_excluded; not_unconditional_action_EV"
MIN_TRAINING_PHYSICAL_MATCHES = 2
FILES = ("fixtures.jsonl", "discovery.jsonl", "attempts.jsonl",
         "context_snapshots.jsonl", "documents.jsonl", "labels/results.jsonl")
NUMERIC_NAMES = (
    ["p_home", "p_draw", "p_away", "prior_home", "prior_draw", "prior_away",
     "normalized_entropy", "hours_to_kickoff_bounded", "known_body_count_bounded"]
    + [f"{channel}:{name}" for channel in
       ("title", "novel_body", "title_entity_counts", "novel_body_entity_counts") for name in FEATURES]
    + ["cost_known", "cost_estimated", "cost_simulated"]
    + ["declared_cost:" + unit for unit in UNITS]
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _probability(value):
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (float, int))
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in value)
            or not math.isclose(sum(value), 1., abs_tol=1e-9, rel_tol=0.)):
        raise ValueError("Expected three finite normalized public probabilities")
    return list(value)


def prequery(raw_menu, action_id):
    """Public function usable at query time; no response/target argument exists.

    URLs bind requests and supply a hostname category, never lexical features.
    Text features use only already-visible parent labels and actual query text.
    Known semantic counts describe f's current input, not the unread article.
    """
    if raw_menu.get("schema") != "prospective-menu-archive-v1":
        raise ValueError("A saved prospective menu is required")
    public = raw_menu["public_before"]
    actions = raw_menu["generated_actions"]
    if len({a["action_id"] for a in actions}) != len(actions):
        raise ValueError("Duplicate pre-query action identity")
    selected = [a for a in actions if a["action_id"] == action_id]
    if len(selected) != 1:
        raise ValueError("Action must occur exactly once in the public menu")
    action = selected[0]
    decision, kickoff = parse_time(public["decision_at"]), parse_time(public["state"]["base"]["kickoff"])
    if (parse_time(raw_menu["observed_at"]) != decision
            or not decision <= parse_time(public["state"]["prediction_cutoff"]) < kickoff):
        raise ValueError("Invalid public pre-query clocks")
    prediction = public["prediction"]
    pipeline = prediction["prediction_pipeline_version"]
    if not isinstance(pipeline, str) or not pipeline:
        raise ValueError("A frozen prediction pipeline is required")
    p = _probability(prediction["probabilities"])
    prior = _probability(prediction["initial_probabilities"])
    projection = prediction["projection"]
    if projection["representation_sha256"] != REPRESENTATION_SHA256:
        raise ValueError("Unknown frozen entity representation")
    # Use the exact candidate-context sidecar frozen with the pre-I/O menu.
    # Re-running the projection here is unsafe: the runtime's allowed-host
    # profile and versioned directory adapter are part of the original
    # experiment, while this consumer must not rediscover or reinterpret them.
    sidecar = public.get("candidate_contexts")
    if not isinstance(sidecar, dict) or set(sidecar) != {
            "by_action_id", "decision_at", "ordering", "schema_version", "semantics"}:
        raise ValueError("A versioned pre-query candidate-context sidecar is required")
    if sidecar.get("schema_version") != "observed-candidate-context-v1":
        raise ValueError("Unsupported candidate-context sidecar schema")
    if parse_time(sidecar.get("decision_at")) != decision:
        raise ValueError("Candidate-context decision clock differs from the frozen menu")
    semantics = sidecar.get("semantics")
    if (not isinstance(semantics, dict)
            or semantics.get("contains_target_body") is not False
            or semantics.get("numerical_value_assigned") is not False):
        raise ValueError("Candidate-context sidecar is not a pre-query-only projection")
    by_action = sidecar.get("by_action_id")
    read_action_ids = {a["action_id"] for a in actions if a.get("tool") == "read_url"}
    if not isinstance(by_action, dict) or set(by_action) != read_action_ids:
        raise ValueError("Candidate-context action coverage differs from the frozen menu")
    for action in actions:
        aid = action["action_id"]
        if action.get("tool") == "search_web":
            continue
        row = by_action[aid]
        if (not isinstance(row, dict) or set(row) != {"action_id", "information_key", "url", "observations"}
                or row["action_id"] != aid or row["information_key"] != action["information_key"]):
            raise ValueError("Candidate-context action identity is not raw-bound")
        if action["tool"] == "read_url" and row["url"] != action.get("url"):
            raise ValueError("Candidate-context URL differs from the frozen action")
        if action["tool"] == "search_web" and row["observations"]:
            raise ValueError("A search action cannot inherit read-url parent observations")
        if not isinstance(row["observations"], list):
            raise ValueError("Candidate-context observations must be a list")
        for observation in row["observations"]:
            if (not isinstance(observation, dict)
                    or set(observation) != {"parent", "link_index", "label_source",
                                             "observed_anchors", "observed_anchors_truncated"}):
                raise ValueError("Candidate-context observation contains undeclared fields")
            parent = observation["parent"]
            if (not isinstance(parent, dict)
                    or set(parent) != {"document_id", "url", "raw_sha256", "observed_at",
                                       "available_at", "source_published_at"}
                    or not isinstance(parent["document_id"], str)
                    or not isinstance(parent["raw_sha256"], str)
                    or len(parent["raw_sha256"]) != 64
                    or parse_time(parent["observed_at"]) > parse_time(parent["available_at"])
                    or parse_time(parent["available_at"]) > decision):
                raise ValueError("Candidate-context parent clock or digest is invalid")
    # This sidecar is the only already-visible text source for the query
    # feature vector. It contains labels from the parent document only.
    rebuilt = sidecar
    texts, labels = [], []
    for aid, context in rebuilt["by_action_id"].items():
        for observation in context["observations"]:
            for anchor in observation["observed_anchors"]:
                text = anchor["text"]
                if text.strip():
                    texts.append(text)
                    if aid == action_id:
                        labels.append(text)
    if action["tool"] == "search_web":
        if not isinstance(action.get("query"), str) or not action["query"].strip():
            raise ValueError("Search query is missing")
        labels.append(action["query"])
        channel = "search_web|" + action["source"]
    elif action["tool"] == "read_url":
        channel = "read_url|" + urlsplit(action["url"]).hostname.lower()
    else:
        raise ValueError("Unsupported acquisition tool")
    numeric = p + prior + [-sum(x * math.log(x) for x in p if x) / math.log(3),
        _bounded((kickoff - decision).total_seconds() / 3600), _bounded(prediction["modeled_read_documents"])]
    for name in ("title", "novel_body", "title_entity_counts", "novel_body_entity_counts"):
        values = projection[name]
        if not isinstance(values, list) or len(values) != len(FEATURES):
            raise ValueError("Frozen entity feature dimensions differ")
        numeric.extend(_bounded(x) for x in values)
    if action["cost_status"] not in {"known", "estimated", "simulated"}:
        raise ValueError("Declared cost status is missing")
    if not {"money", "requests"} <= set(action["cost"]):
        raise ValueError("Declared resource cost is missing")
    numeric += [float(action["cost_status"] == x) for x in ("known", "estimated", "simulated")]
    numeric += [_bounded(action["cost"].get(unit, 0.)) for unit in UNITS]
    sources = sorted({urlsplit(d["url"]).hostname for d in public["documents"]
                      if d.get("status") == "OK"})
    context = {"texts": sorted(set(texts)), "labels": sorted(set(labels)), "numeric": numeric,
        "sources": sources, "channel": channel, "mask": [bool(texts), bool(labels)],
        "decision_at": decision.isoformat(), "kickoff_at": kickoff.isoformat(),
        "date": decision.astimezone(timezone.utc).date().isoformat()}
    payload = {"schema": QUERY_VERSION, "prediction_pipeline_version": pipeline,
        "predictor_model_version": prediction["model_version"], "action_id": action_id,
        "source_match_id": public["state"]["match_id"], "numeric_names": NUMERIC_NAMES,
        "feature_origin": "derived_from_saved_pre_io_public_menu_not_online_g_evaluation",
        "context": context}
    return {**payload, "query_sha256": _digest(payload)}


def _check_query(query):
    if (query.get("schema") != QUERY_VERSION or query.get("numeric_names") != NUMERIC_NAMES
            or query.get("query_sha256") != _digest({k: v for k, v in query.items() if k != "query_sha256"})):
        raise ValueError("Invalid or modified g query")
    c = query["context"]
    if (len(c["numeric"]) != len(NUMERIC_NAMES)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in c["numeric"])):
        raise ValueError("Invalid pre-query numeric vector")


def assemble(entries):
    """Re-audit source streams and attach immutable, preassigned group slots.

    Each entry has stream and registration paths. Labels are recomputed using
    the existing raw-bound join, never trusted from a cached derived JSONL.
    """
    records, bindings, partitions, slots, seen = [], [], {}, {}, set()
    for entry in entries:
        root, registration = Path(entry["stream"]).resolve(), Path(entry["registration"]).resolve()
        reg = _read(registration)
        if reg.get("schema") != "footballmind-cumulative-shadow-slot-v2":
            raise ValueError("Expected a pre-registered cumulative shadow slot v2")
        mid, partition = reg["physical_match_id"], reg["partition"]
        if partition not in {"train", "holdout"} or not isinstance(mid, str) or not mid:
            raise ValueError("Invalid canonical group or partition")
        if mid in partitions and partitions[mid] != partition:
            raise ValueError("A physical match cannot cross train and holdout")
        slot_key = (reg["cohort_id"], reg["slot"])
        if slot_key in slots and slots[slot_key] != mid:
            raise ValueError("One pre-registered slot cannot represent two physical matches")
        slots[slot_key], partitions[mid] = mid, partition
        data = {name: _rows(root / name) for name in FILES}
        fixtures = data["fixtures.jsonl"]
        if len(fixtures) != 1 or fixtures[0]["physical_match_id"] != reg["fixture_id"]:
            raise ValueError("Registration must bind the exact single-fixture stream")
        if parse_time(fixtures[0]["kickoff_at"]) != parse_time(reg["kickoff"]):
            raise ValueError("Registered kickoff differs from source fixture")
        fixture = fixtures[0]
        menus = {m["discovery_id"]: m for m in data["discovery.jsonl"]}
        attempts = {a["attempt_id"]: a for a in data["attempts.jsonl"] if a.get("selected_by_policy") is True}
        if len(menus) != len(data["discovery.jsonl"]):
            raise ValueError("Duplicate discovery identity")
        if len(attempts) != sum(a.get("selected_by_policy") is True for a in data["attempts.jsonl"]):
            raise ValueError("Duplicate selected attempt identity")
        derived = derive_g_candidates(fixtures=fixtures, discoveries=data["discovery.jsonl"],
            attempts=data["attempts.jsonl"], contexts=data["context_snapshots.jsonl"],
            documents=data["documents.jsonl"], results=data["labels/results.jsonl"], archive_root=root)
        for row in derived:
            aid = row["attempt_id"]
            if aid in seen:
                raise ValueError("Duplicate actual attempt across input streams")
            seen.add(aid)
            attempt, query, feature_error = attempts[aid], None, None
            if not parse_time(reg["registered_at"]) < parse_time(attempt["decision_at"]):
                raise ValueError("Match split was not registered before acquisition")
            if parse_time(attempt["decision_at"]).astimezone(timezone.utc).date().isoformat() != reg["decision_date_utc"]:
                raise ValueError("Registered UTC decision date differs")
            quality = row.get("g_quality_target") or {}
            if quality.get("prediction_pipeline_version") != reg["prediction_pipeline_version"]:
                raise ValueError("Registration and target belong to different prediction pipelines")
            try:
                menu = menus[attempt["discovery_id"]]
                raw = _payload(root, menu)
                query = prequery(raw, row["action_id"])
                if (query["source_match_id"] != reg["fixture_id"]
                        or query["prediction_pipeline_version"] != reg["prediction_pipeline_version"]
                        or parse_time(query["context"]["decision_at"]) != parse_time(attempt["decision_at"])):
                    raise ValueError("Query registration or decision binding differs")
            except (KeyError, TypeError, ValueError) as exc:
                feature_error = str(exc)
            reasons = list(quality.get("exclusion_reason", []))
            if (quality.get("schema") != "frozen-predictor-quality-target-v1"
                    or quality.get("target") != TARGET or quality.get("scope") != SCOPE
                    or quality.get("conditioning") != CONDITIONING or quality.get("cost_subtracted") is not False):
                reasons.append("unsupported_quality_target_contract")
            if feature_error:
                reasons.append("prequery_features_unavailable")
            eligible = quality.get("status") == "eligible" and not reasons
            records.append({"sample_id": aid, "physical_match_id": mid, "partition": partition,
                "source_match_id": reg["fixture_id"], "pipeline": reg["prediction_pipeline_version"],
                "query": query, "feature_error": feature_error, "selection_probability": row["selection_probability"],
                "target_status": "eligible" if eligible else "ineligible",
                "gain": quality.get("realized_signed_gain") if eligible else None,
                "exclusion_reason": sorted(set(reasons)), "predictor_pair_audit": row.get("predictor_pair_audit"),
                "source_stream": str(root), "outcome": row["outcome"]})
        bindings.append({"stream": str(root), "registration": str(registration),
            "registration_sha256": sha256(registration.read_bytes()).hexdigest(),
            "stream_files": {name: sha256((root / name).read_bytes()).hexdigest() for name in FILES},
            "physical_match_id": mid, "partition": partition, "pipeline": reg["prediction_pipeline_version"]})
    summary = {"selected_attempts": len(records), "physical_matches": len(partitions),
        "eligible_rows": sum(r["target_status"] == "eligible" for r in records),
        "prequery_feature_rows": sum(r["query"] is not None for r in records),
        "valid_predictor_pairs": sum(bool((r["predictor_pair_audit"] or {}).get("valid")) for r in records),
        "exclusions": dict(Counter(reason for r in records for reason in r["exclusion_reason"]))}
    return {"schema": DATA_VERSION, "records": records, "summary": summary,
        "source_bindings": bindings, "partitions": partitions,
        "network_requests": 0, "production_ready": False, "selection_allowed": False}


class ProspectiveQualityG:
    """Versioned research estimator; cannot be loaded as the live g policy."""
    def __init__(self):
        self.kernel, self.metadata, self.model_version = ContextValueModel(), {}, ""

    def fit(self, rows):
        if not rows or any(r["partition"] != "train" or r["target_status"] != "eligible" for r in rows):
            raise ValueError("Only admitted training-group rows can fit g")
        training_match_ids = {r["physical_match_id"] for r in rows}
        if len(training_match_ids) < MIN_TRAINING_PHYSICAL_MATCHES:
            raise ValueError("At least two independent training physical matches are required")
        pipelines = {r["pipeline"] for r in rows}
        if len(pipelines) != 1:
            raise ValueError("Do not pool different f/extraction pipelines")
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate fitting samples")
        gains = [r["gain"] for r in rows]
        if (any(isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) for g in gains)
                or np.ptp(gains) <= 1e-12):
            raise ValueError("No variable admitted target; retain a constant baseline only")
        kernel_rows = []
        for r in rows:
            _check_query(r["query"])
            if r["query"]["prediction_pipeline_version"] != r["pipeline"]:
                raise ValueError("Query and target pipeline differ")
            kernel_rows.append({"physical_match_id": r["physical_match_id"],
                "context": r["query"]["context"], "gain": r["gain"]})
        counts, objective = self.kernel._fit_regression(kernel_rows)
        baseline = float(np.mean([np.mean([r["gain"] for r in rows if r["physical_match_id"] == mid]) for mid in counts]))
        self.metadata = {"prediction_pipeline_version": next(iter(pipelines)), "target": TARGET,
            "conditioning": CONDITIONING, "scope": SCOPE, "cost_subtracted": False,
            "training_physical_match_ids": sorted(counts), "training_rows": len(rows),
            "training_rows_sha256": _digest(rows), "constant_baseline": baseline,
            "fit_execution_at": datetime.now(timezone.utc).isoformat(), "objective": objective,
            "weighting": "equal_physical_match_then_equal_observed_successful_transition",
            "propensity_role": "logged_only_not_off_policy_or_counterfactual_evaluation",
            "encoder_fit": "training_group_only; no_holdout_vocabulary_or_statistics",
            "query_time_information_only": True, "evidence_origin": "prospective_capture",
            "uncertainty": None, "risk_of_harm": None, "calibration_status": "not_established",
            "production_ready": False, "selection_allowed": False}
        self.model_version = VERSION + ":" + _digest(self.payload())[:24]
        return self

    def payload(self):
        return {"schema": VERSION, "query_schema": QUERY_VERSION, "parameters": PARAMETERS,
            "metadata": self.metadata, "encoder": self.kernel.encoder,
            "coefficients": self.kernel.coefficients, "support_records": self.kernel.support_records}

    def estimate(self, query):
        if self.model_version != VERSION + ":" + _digest(self.payload())[:24]:
            raise ValueError("Frozen g model has changed")
        _check_query(query)
        if query["prediction_pipeline_version"] != self.metadata["prediction_pipeline_version"]:
            raise ValueError("Query belongs to a different prediction pipeline")
        vector, _ = self.kernel._vector(query["context"])
        return {"research_expected_signed_gain": float(vector @ np.asarray(self.kernel.coefficients)),
            "support_status": "research_estimate_not_calibrated", "conditioning": CONDITIONING,
            "expected_net_voi": None, "risk_of_harm": None, "selection_allowed": False,
            "production_ready": False, "model_version": self.model_version}

    def save(self, path):
        if self.model_version != VERSION + ":" + _digest(self.payload())[:24]:
            raise ValueError("Invalid frozen g model")
        with Path(path).open("x", encoding="utf-8") as out:
            json.dump({**self.payload(), "model_version": self.model_version}, out, ensure_ascii=False,
                      indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")

    @classmethod
    def load(cls, path):
        value, obj = _read(path), cls()
        if value.get("schema") != VERSION or value.get("parameters") != PARAMETERS or value.get("query_schema") != QUERY_VERSION:
            raise ValueError("Unsupported quality g model")
        obj.metadata, obj.model_version = value["metadata"], value["model_version"]
        for key in ("encoder", "coefficients", "support_records"):
            setattr(obj.kernel, key, value[key])
        if obj.model_version != VERSION + ":" + _digest(obj.payload())[:24]:
            raise ValueError("Quality g artifact digest differs")
        return obj


def fit_and_evaluate(dataset):
    """Fit separately by pipeline; preserve holdout even when train is empty."""
    if dataset.get("schema") != DATA_VERSION:
        raise ValueError("Expected an assembled prospective quality dataset")
    strata, models, reports, seen = defaultdict(list), {}, {}, set()
    groups = {}
    for r in dataset["records"]:
        if r["sample_id"] in seen:
            raise ValueError("Duplicate actual attempt")
        seen.add(r["sample_id"])
        if r["partition"] not in {"train", "holdout"}:
            raise ValueError("Unknown partition")
        if r["physical_match_id"] in groups and groups[r["physical_match_id"]] != r["partition"]:
            raise ValueError("Physical match overlaps training and holdout")
        groups[r["physical_match_id"]] = r["partition"]
        strata[r["pipeline"]].append(r)
    for pipeline, records in strata.items():
        train = [r for r in records if r["partition"] == "train" and r["target_status"] == "eligible"]
        test = [r for r in records if r["partition"] == "holdout" and r["target_status"] == "eligible"]
        report = {"train_rows": len(train), "holdout_rows": len(test),
            "train_physical_matches": len({r["physical_match_id"] for r in train}),
            "holdout_physical_matches": len({r["physical_match_id"] for r in test}),
            "pending_or_excluded_rows": len(records) - len(train) - len(test),
            "production_ready": False, "selection_allowed": False,
            "fit_status": "waiting_for_admitted_training_targets", "heldout": None}
        if train and len({r["physical_match_id"] for r in train}) < MIN_TRAINING_PHYSICAL_MATCHES:
            report.update(fit_status="insufficient_training_physical_matches",
                          minimum_training_physical_matches=MIN_TRAINING_PHYSICAL_MATCHES)
        elif train and np.ptp([r["gain"] for r in train]) <= 1e-12:
            report.update(fit_status="constant_target_only_no_conditional_g_fit", constant_baseline=train[0]["gain"])
        elif train:
            model = ProspectiveQualityG().fit(train)
            models[pipeline] = model
            report.update(fit_status="fitted_research_only", model_version=model.model_version,
                          constant_baseline=model.metadata["constant_baseline"])
            if test:
                predictions = [{"sample_id": r["sample_id"], "physical_match_id": r["physical_match_id"],
                    "observed_gain": r["gain"], **model.estimate(r["query"])} for r in test]
                by_match = defaultdict(list)
                for r in predictions:
                    by_match[r["physical_match_id"]].append(r)
                mse = lambda field: float(np.mean([np.mean([(field(r) - r["observed_gain"]) ** 2 for r in group])
                                                   for group in by_match.values()]))
                report["heldout"] = {"predictions": predictions,
                    "equal_match_mse": mse(lambda r: r["research_expected_signed_gain"]),
                    "train_constant_equal_match_mse": mse(lambda r: model.metadata["constant_baseline"]),
                    "scope": "observed_successful_action_labels_only_not_policy_value_or_top1_accuracy"}
        reports[pipeline] = report
    return models, {"schema": VERSION, "strata": reports, "models_fitted": len(models),
        "evaluation_unit": "independent_physical_match_within_exact_prediction_pipeline",
        "network_requests": 0, "selection_allowed": False, "production_ready": False}
