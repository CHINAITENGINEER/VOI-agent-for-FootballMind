"""Outcome and derived-sample projection for the prospective stream.

This module is deliberately downstream of acquisition.  It does not fetch a
result and it does not write to a run's immutable files.  Independent result
rows produced by the existing :mod:`verified_fixtures`/``settlement`` gates
can be projected into ``labels/results.jsonl`` in a new stream package, while
action records can be joined to those labels to make auditable (and usually
ineligible) g-training candidates.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import hashlib
import json
from math import fsum, isfinite, log
from pathlib import Path
import re
import shutil

from football_spider.storage.raw_archive import RawArchive
from football_platform.information.stream_validator import _safe_raw
from .predictor_context_pairs import is_predictor_context, verify_predictor_pair

_MANIFEST_OBSERVATION = re.compile(r"(?:^|\|)observation:([^|]+)")


def _time(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo and parsed.utcoffset() is not None else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_for(row: Mapping, archive: RawArchive):
    """Resolve a settlement row's declared observation without guessing."""
    for value in row.get("input_manifest", ()) or ():
        match = _MANIFEST_OBSERVATION.search(str(value))
        if not match:
            continue
        found = [item for item in archive.observations() if item.id == match.group(1)]
        if len(found) == 1:
            return found[0]
    return None


def project_result_rows(settlement_rows: Iterable[Mapping], *, physical_match_ids: Mapping[str, str] | None = None,
                        result_archive: str | Path | None = None, output_root: str | Path | None = None):
    """Project verified settlement rows into canonical result rows.

    ``settlement_rows`` must come from the existing independent settlement
    gateway.  A row with no final regulation score is rejected.  If an output
    package and archive are supplied, the declared RawArchive observation is
    copied byte-for-byte into ``raw/results`` and the returned row points at
    that copy.  No source archive is modified.
    """
    mapping = dict(physical_match_ids or {})
    archive = RawArchive(Path(result_archive)) if result_archive is not None else None
    package = Path(output_root) if output_root is not None else None
    rows, rejected = [], []
    for source in settlement_rows:
        if not isinstance(source, Mapping):
            rejected.append({"reason": "result_row_not_object"})
            continue
        match_id = source.get("match_id") or source.get("physical_match_id")
        home, away = source.get("home_goals"), source.get("away_goals")
        observed = source.get("observed_at") or source.get("result_observed_at")
        kickoff = _time(source.get("kickoff") or source.get("kickoff_at"))
        gateway_verified = (source.get("independent_outcome_verified") is True
                            or (isinstance(source.get("verification_schema"), str)
                                and isinstance(source.get("identity_binding_sha256"), str)
                                and isinstance(source.get("regulation_rule_version"), str)))
        if (not isinstance(match_id, str) or type(home) is not int or home < 0
                or type(away) is not int or away < 0 or _time(observed) is None
                or kickoff is None or not _time(observed) > kickoff
                or source.get("status") not in {"final", "FINISHED", "finished"}
                or source.get("score_period", "regulation") != "regulation"
                or not gateway_verified):
            rejected.append({"match_id": match_id, "reason": "missing_final_regulation_score_or_clock"})
            continue
        physical = mapping.get(match_id, source.get("physical_match_id", match_id))
        raw_path = source.get("raw_path")
        raw_hash = source.get("raw_sha256")
        observation = None
        if archive is not None:
            observation = _observation_for(source, archive)
            if observation is None:
                rejected.append({"match_id": match_id, "reason": "declared_result_observation_missing"})
                continue
            try:
                # RawArchive.read_body verifies the compressed payload against
                # its immutable observation digest before any copy/reference.
                archive.read_body(observation)
            except (OSError, ValueError, EOFError):
                rejected.append({"match_id": match_id, "reason": "result_raw_digest_mismatch"})
                continue
            if package is not None:
                target = package / "raw" / "results" / (observation.id + ".body.gz")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(archive.root / observation.payload_path, target)
                raw_path = target.relative_to(package).as_posix()
                raw_hash = observation.content_hash
            else:
                raw_path = observation.payload_path
                raw_hash = observation.content_hash
        if archive is None:
            # A declaration without the source archive cannot prove bytes.
            rejected.append({"match_id": match_id, "reason": "result_archive_required_for_digest"})
            continue
        if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_hash, str):
            rejected.append({"match_id": match_id, "reason": "result_raw_evidence_missing"})
            continue
        result = "home" if home > away else "draw" if home == away else "away"
        source_url = source.get("source_url")
        if not isinstance(source_url, str) or not source_url:
            source_url = observation.endpoint if observation is not None else source.get("source") or "unknown"
        rows.append({
            "physical_match_id": physical,
            "regulation_home_goals": home,
            "regulation_away_goals": away,
            "result_status": "final",
            "result_source_url": source_url,
            "result_published_at": source.get("source_published_at"),
            "result_observed_at": observed,
            "raw_path": raw_path,
            "raw_sha256": raw_hash,
            "timing_basis": source.get("observation_time_basis") or "independent_result_archive_observed_at",
            "result": result,
            "information_mode": "post_match_outcome",
            "independent_outcome_verified": True,
        })
    return rows, {"accepted": len(rows), "rejected": rejected}


def _number(value, *, positive=False):
    return type(value) in {int, float} and isfinite(value) and (value > 0 if positive else value >= 0)


def _groups(values, field):
    grouped = {}
    for row in values:
        if isinstance(row, Mapping):
            grouped.setdefault(row.get(field), []).append(row)
    return grouped


def _verified_raw(row, root):
    """Reuse the stream validator's byte-level RawArchive/path/hash check."""
    if root is None:
        return False, ["archive_root_missing"]
    issues = []
    try:
        _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), "raw_path", issues)
    except (OSError, ValueError, EOFError):
        # Damaged compressed input is an exclusion, never a reason to trust
        # the digest declaration or to abort all unrelated selected actions.
        return False, ["raw_read_error"]
    return not issues, [issue.code for issue in issues]


def _normalized_prediction(value):
    if not isinstance(value, Mapping) or set(value) != {"home", "draw", "away"}:
        return None
    values = [value[key] for key in ("home", "draw", "away")]
    if not all(_number(x) and x <= 1 for x in values) or abs(fsum(values) - 1) > 1e-8:
        return None
    return values


def derive_g_candidates(*, fixtures: Iterable[Mapping], discoveries: Iterable[Mapping], attempts: Iterable[Mapping],
                        contexts: Iterable[Mapping], results: Iterable[Mapping], archive_root: str | Path | None = None,
                        documents: Iterable[Mapping] | None = None):
    """Join canonical rows using actual archived bytes and strict time gates.

    ``archive_root`` is the root of the exported stream package.  Omitting it
    remains supported, but every row is ineligible because a declared digest
    is not byte verification.  Fixture, menu, action and result evidence all
    use the existing stream validator's RawArchive-aware SHA-256 checks.
    Missing clocks, predictions, propensities and costs remain null; no
    normalization, timestamp fallback or outcome inference repairs evidence.
    A declared returned document additionally requires a unique canonical
    document row, its measured capture and Agent availability clocks, and
    verified raw bytes.  Omitting ``documents`` preserves actions without a
    returned document, but cannot prove a declared document's provenance.
    """
    root = Path(archive_root).expanduser().resolve() if archive_root is not None else None
    fixture_groups = _groups(fixtures, "physical_match_id")
    menu_groups = _groups(discoveries, "discovery_id")
    result_groups = _groups(results, "physical_match_id")
    document_groups = _groups(documents if documents is not None else (), "document_id")
    context_rows = [row for row in contexts if isinstance(row, Mapping)]
    before = _groups((r for r in context_rows if r.get("snapshot_kind") == "before_action"), "physical_match_id")
    after = _groups((r for r in context_rows if r.get("snapshot_kind") == "after_action"), "parent_attempt_id")
    rows = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or attempt.get("selected_by_policy") is not True:
            continue
        match_id, aid = attempt.get("physical_match_id"), attempt.get("attempt_id")
        reasons = []

        def one(grouped, key, name):
            values = grouped.get(key, [])
            if len(values) != 1:
                reasons.append(name + ("_missing" if not values else "_ambiguous"))
                return {}
            return values[0]

        fixture = one(fixture_groups, match_id, "fixture")
        menu = one(menu_groups, attempt.get("discovery_id"), "menu")
        result = one(result_groups, match_id, "outcome")
        a = one(after, aid, "after_context")
        # v3 separates forecast information time from later archive handoff.
        # Bind exactly this menu's forecast, never a previous cycle's snapshot.
        decision_probe = _time(attempt.get("decision_at"))
        predictor_pair_mode = (is_predictor_context(a) or any(is_predictor_context(row)
            for row in before.get(match_id, [])))
        before_candidates = [row for row in before.get(match_id, []) if (
            is_predictor_context(row) and decision_probe is not None
            and _time(row.get("forecast_as_of")) == decision_probe) ] if predictor_pair_mode else [
                row for row in before.get(match_id, []) if _time(row.get("observed_at")) is not None
                and (decision_probe is None or _time(row.get("observed_at")) <= decision_probe)]
        if predictor_pair_mode and len(before_candidates) != 1:
            b = {}
            reasons.append("before_predictor_context_missing" if not before_candidates else "before_predictor_context_ambiguous")
        elif before_candidates:
            b = max(before_candidates, key=lambda row: _time(row.get("observed_at")))
        else:
            b = {}
            reasons.append("before_context_missing")
        predictor_pair = (verify_predictor_pair(before=b, after=a, menu=menu, attempt=attempt,
            fixture=fixture, root=root) if predictor_pair_mode else None)
        if predictor_pair is not None:
            reasons.extend(predictor_pair["reasons"])
        # A run that declares a fitted f must prove the selected document was
        # part of the immutable after-state input. A changed (or unchanged)
        # probability is not evidence of consumption. Legacy runs without a
        # predictor declaration keep their historical eligibility behavior.
        f_consumption_required = attempt.get("f_consumption_required") is True
        f_consumption_proven = (predictor_pair is not None
                                and predictor_pair.get("valid") is True)
        if f_consumption_required and not f_consumption_proven:
            reasons.append("f_consumption_not_proven")
        kickoff = _time(fixture.get("kickoff_at"))

        def pre(value):
            parsed = _time(value)
            return parsed is not None and kickoff is not None and parsed < kickoff

        if fixture.get("information_mode") != "prospective_capture":
            reasons.append("historical_reconstruction")
        if not (pre(fixture.get("registered_at")) and pre(fixture.get("arm_assigned_at"))
                and fixture.get("pre_kickoff_frozen") is True):
            reasons.append("prospective_registration_not_proven")
        arm = attempt.get("policy_arm")
        if (arm not in {"voi", "fixed_order", "random_explore", "full_probe"}
                or fixture.get("policy_arm") != arm or not attempt.get("policy_version")
                or fixture.get("policy_version") != attempt.get("policy_version")
                or menu.get("policy_version") != attempt.get("policy_version")):
            reasons.append("policy_assignment_not_proven")
        if arm == "random_explore" and type(fixture.get("random_seed")) is not int:
            reasons.append("random_seed_missing")
        original_menu = menu.get("original_agent_menu_proven") is True
        if not original_menu:
            reasons.append("original_agent_menu_not_proven")
        if menu.get("information_mode") != "prospective_capture":
            reasons.append("menu_not_prospective")
        candidate_rows = menu.get("visible_candidates")
        candidates = [c for c in candidate_rows if isinstance(c, Mapping)
                      and c.get("candidate_id") == attempt.get("candidate_id")] if isinstance(candidate_rows, list) else []
        if (menu.get("physical_match_id") != match_id or len(candidates) != 1
                or candidates[0].get("action_id", attempt.get("action_id")) != attempt.get("action_id")):
            reasons.append("candidate_menu_binding_not_proven")
        decision = _time(attempt.get("decision_at"))
        requested = _time(attempt.get("requested_at"))
        started = _time(attempt.get("request_started_at", attempt.get("requested_at")))
        response = _time(attempt.get("response_observed_at"))
        action_pre = all(value is not None and kickoff is not None and value < kickoff
                         for value in (decision, requested, started, response))
        if not action_pre:
            reasons.append("action_time_unknown_or_after_kickoff")
        if not (decision is not None and requested is not None and started is not None
                and response is not None and decision <= requested <= started <= response):
            reasons.append("action_time_order_invalid")
        if attempt.get("status") in {"not_requested", "not_configured", "unknown", None}:
            reasons.append("action_execution_not_proven")
        if attempt.get("information_mode", "prospective_capture") != "prospective_capture":
            reasons.append("action_not_prospective")
        menu_time = _time(menu.get("observed_at"))
        menu_pre = pre(menu.get("observed_at"))
        if not menu_pre:
            reasons.append("menu_time_unknown_or_after_kickoff")
        if menu_time is None or decision is None or menu_time > decision:
            reasons.append("menu_after_decision_or_unknown")

        raw_checks, raw_details = {}, {}
        for name, source in (("fixture", fixture), ("menu", menu), ("action", attempt), ("outcome", result)):
            raw_checks[name], raw_details[name] = _verified_raw(source, root)
            if not raw_checks[name]:
                reasons.append(name + "_raw_digest_unverified")
        probability = attempt.get("selection_probability")
        propensity = _number(probability, positive=True) and probability <= 1
        if not propensity:
            reasons.append("propensity_missing" if probability is None else "propensity_invalid")
        if (not attempt.get("selection_probability_basis")
                or attempt.get("selection_probability_basis") in {"not_recorded", "not_configured"}
                or attempt.get("selection_basis") in {None, "manual_predeclared", "not_configured"}):
            reasons.append("propensity_basis_not_proven")

        context_pre = True
        for label, state in (("before", b), ("after", a)):
            if state.get("physical_match_id") != match_id or state.get("information_mode") != "prospective_capture":
                reasons.append(label + "_context_binding_or_mode_invalid")
            clocks = {key: _time(state.get(key)) for key in
                      ("observed_at", "captured_at", "available_at", "prediction_cutoff")}
            valid = all(stamp is not None and kickoff is not None and stamp < kickoff for stamp in clocks.values())
            context_pre = context_pre and valid
            if not valid:
                reasons.append(label + "_context_time_unknown_or_after_kickoff")
            if not state.get("availability_basis") or state.get("availability_status") == "unknown":
                reasons.append(label + "_availability_not_proven")
            observed, captured, available, cutoff = [clocks[x] for x in
                ("observed_at", "captured_at", "available_at", "prediction_cutoff")]
            if not predictor_pair_mode and not (observed is not None and captured is not None and available is not None and cutoff is not None
                    and observed <= captured and observed <= available <= cutoff):
                reasons.append(label + "_context_time_order_invalid")
            if label == "before" and not (decision is not None and cutoff is not None and cutoff <= decision):
                reasons.append("before_context_after_decision_or_unknown")
            if label == "after" and not (response is not None and observed is not None and response <= observed):
                reasons.append("after_context_before_response_or_unknown")
            if state.get("raw_path") is not None or state.get("raw_sha256") is not None:
                raw_checks[label + "_context"], raw_details[label + "_context"] = _verified_raw(state, root)
                if not raw_checks[label + "_context"]:
                    reasons.append(label + "_context_raw_digest_unverified")
        if predictor_pair is not None:
            raw_checks["predictor_pair"] = predictor_pair["valid"]
            raw_details["predictor_pair"] = predictor_pair["reasons"]
            context_pre = context_pre and all(predictor_pair["context_time_proven"].values())

        returned_document_id = attempt.get("returned_document_id")
        document_required = returned_document_id is not None
        document_capture_proven = document_availability_proven = None
        if document_required:
            if not isinstance(returned_document_id, str) or not returned_document_id.strip():
                reasons.append("document_id_invalid")
                document = {}
            else:
                document = one(document_groups, returned_document_id, "document")
            document_match_ids = document.get("physical_match_ids")
            declared_match_id = document.get("physical_match_id")
            bound = (declared_match_id == match_id if declared_match_id is not None
                     else isinstance(document_match_ids, list) and match_id in document_match_ids)
            if not bound:
                reasons.append("document_fixture_binding_not_proven")
            if document.get("information_mode") != "prospective_capture":
                reasons.append("document_not_prospective")
            observed_document = _time(document.get("observed_at"))
            captured_document = _time(document.get("captured_at"))
            available_document = _time(document.get("available_at"))
            observed_pre = pre(document.get("observed_at"))
            captured_pre = pre(document.get("captured_at"))
            available_pre = pre(document.get("available_at"))
            for name, proven in (("observed", observed_pre), ("capture", captured_pre),
                                 ("availability", available_pre)):
                if not proven:
                    reasons.append("document_" + name + "_time_unknown_or_after_kickoff")
            measured_capture = (document.get("capture_status") == "known"
                                and document.get("capture_timing_basis") in {
                                    "measured_local_archive_capture",
                                    "measured_local_utc_archive_save_return"})
            capture_order = (observed_document is not None and captured_document is not None
                             and observed_document <= captured_document)
            document_capture_proven = observed_pre and captured_pre and measured_capture and capture_order
            if not measured_capture:
                reasons.append("document_capture_not_proven")
            if not capture_order:
                reasons.append("document_capture_time_order_invalid")
            availability_basis = document.get("availability_basis")
            availability_recorded = (isinstance(availability_basis, str) and bool(availability_basis.strip())
                                     and availability_basis not in {"unknown", "not_recorded", "not_configured"}
                                     and document.get("availability_status") == "known")
            availability_order = (observed_document is not None and available_document is not None
                                  and observed_document <= available_document)
            after_cutoff = _time(a.get("prediction_cutoff"))
            available_before_prediction = (available_document is not None and after_cutoff is not None
                                           and available_document <= after_cutoff)
            if predictor_pair is not None:
                # The canonical document available_at is a later context
                # handoff. Its earlier projector clock must be proven by the
                # hashed f input, never inferred from the HTTP arrival alone.
                input_document = predictor_pair.get("document_input") or {}
                input_available = _time(input_document.get("available_at"))
                available_before_prediction = bool(predictor_pair["valid"]
                    and input_document.get("document_id") == returned_document_id
                    and input_document.get("raw_sha256") == document.get("raw_sha256") == attempt.get("raw_sha256")
                    and input_available is not None and captured_document is not None and after_cutoff is not None
                    and captured_document <= input_available <= after_cutoff
                    and _time(document.get("available_at")) == _time(a.get("available_at")))
            document_availability_proven = (observed_pre and available_pre and availability_recorded
                                            and availability_order and available_before_prediction)
            if not availability_recorded:
                reasons.append("document_availability_not_proven")
            if "document_admitted" in document and document.get("document_admitted") is not True:
                reasons.append("document_not_admitted")
                document_availability_proven = False
            if not availability_order:
                reasons.append("document_availability_time_order_invalid")
            if not available_before_prediction:
                reasons.append("document_available_after_prediction_or_unknown")
            raw_checks["document"], raw_details["document"] = _verified_raw(document, root)
            if not raw_checks["document"]:
                reasons.append("document_raw_digest_unverified")

        observed_result = _time(result.get("result_observed_at"))
        outcome_after = observed_result is not None and kickoff is not None and observed_result > kickoff
        if not outcome_after:
            reasons.append("outcome_time_unknown_or_before_kickoff")
        independent = result.get("independent_outcome_verified") is True
        if not independent:
            reasons.append("independent_outcome_not_verified")
        if result.get("result_status") != "final" or result.get("score_period", "regulation") != "regulation":
            reasons.append("final_regulation_outcome_not_proven")
        if result.get("information_mode") != "post_match_outcome" or not result.get("result_source_url"):
            reasons.append("outcome_provenance_not_proven")
        if result.get("kickoff_at") is not None and _time(result["kickoff_at"]) != kickoff:
            reasons.append("outcome_fixture_time_mismatch")
        home, away = result.get("regulation_home_goals"), result.get("regulation_away_goals")
        outcome = None
        if type(home) is int and home >= 0 and type(away) is int and away >= 0:
            outcome = "home" if home > away else "draw" if home == away else "away"
            if result.get("result", outcome) != outcome:
                reasons.append("outcome_score_conflict")
        else:
            reasons.append("regulation_score_invalid")
        prediction_before, prediction_after = b.get("prediction_before"), a.get("prediction_after")
        pb, pa = _normalized_prediction(prediction_before), _normalized_prediction(prediction_after)
        if prediction_before is None or prediction_after is None:
            reasons.append("prediction_missing")
        elif pb is None or pa is None:
            reasons.append("prediction_invalid")
        # g estimates quality improvement; resource prices are a separate
        # decision input. New asynchronous f pairs expose a gross log-loss
        # target even when actual monetary cost is unknown. Keep the legacy
        # net-gain label and its cost gate unchanged for existing consumers.
        quality_target = None
        if (b.get("raw_context_schema") == "runtime-predictor-context-v4"
                and a.get("raw_context_schema") == "runtime-predictor-context-v4"):
            quality_reasons = list(reasons)
            gain = None
            if not quality_reasons and outcome is not None and pb is not None and pa is not None:
                yi = {"home": 0, "draw": 1, "away": 2}[outcome]
                if pb[yi] == 0 or pa[yi] == 0:
                    quality_reasons.append("prediction_zero_for_outcome")
                else:
                    gain = log(pa[yi]) - log(pb[yi])
            quality_target = {
                "schema": "frozen-predictor-quality-target-v1", "status": "eligible" if not quality_reasons else "ineligible",
                "target": "conditional_expected_gross_log_loss_reduction",
                "conditioning": "both_f_evaluations_completed_on_time; extraction_failures_excluded; not_unconditional_action_EV",
                "realized_signed_gain": gain,
                "prediction_pipeline_version": b.get("prediction_pipeline_version"),
                "unit": "natural_log_loss", "cost_subtracted": False,
                "scope": "entire_action_visible_state_change_including_exposed_links_and_body",
                "production_ready": False, "exclusion_reason": sorted(set(quality_reasons))}
        losses = {"loss_before": None, "loss_after": None, "realized_gain": None}
        cost = attempt.get("acquisition_cost", attempt.get("monetary_cost_if_known"))
        if cost is None:
            reasons.append("acquisition_cost_unknown")
        elif not _number(cost):
            reasons.append("acquisition_cost_invalid")
        # A returned score alone is not a training label.  Compute derived
        # values only after all provenance, time and independent-outcome gates.
        if not reasons and outcome is not None and pb is not None and pa is not None:
            index = {"home": 0, "draw": 1, "away": 2}[outcome]
            if pb[index] == 0 or pa[index] == 0:
                reasons.append("prediction_zero_for_outcome")
            else:
                losses["loss_before"], losses["loss_after"] = -log(pb[index]), -log(pa[index])
                losses["realized_gain"] = losses["loss_before"] - losses["loss_after"]
        rows.append({"physical_match_id": match_id, "attempt_id": aid, "action_id": attempt.get("action_id"),
            "policy_arm": arm, "selection_probability": probability,
            "f_consumption_required": f_consumption_required,
            "f_consumption_status": attempt.get("f_consumption_status"),
            "f_consumed_input_ids": attempt.get("f_consumed_input_ids"),
            "f_consumption_proven": f_consumption_proven if f_consumption_required else None,
            "prediction_before": prediction_before, "prediction_after": prediction_after, "outcome": outcome,
            **losses, "acquisition_cost": cost,
            "net_gain": losses["realized_gain"] - cost if losses["realized_gain"] is not None and _number(cost) else None,
            "label_status": "eligible" if not reasons else "ineligible",
            "eligibility_checks": {"menu_before_kickoff": menu_pre, "action_before_kickoff": action_pre,
                "context_before_kickoff": context_pre, "outcome_after_kickoff": outcome_after,
                "independent_outcome_verified": independent, "original_agent_menu_proven": original_menu,
                "document_evidence_required": document_required,
                "document_capture_proven": document_capture_proven,
                "document_availability_proven": document_availability_proven,
                "f_consumption_required": f_consumption_required,
                "f_consumption_proven": f_consumption_proven if f_consumption_required else None,
                "raw_digest_verified": all(raw_checks.values()), "propensity_available": propensity},
            "raw_digest_checks": raw_checks, "raw_digest_errors": raw_details,
            **({"predictor_pair_audit": {key: value for key, value in predictor_pair.items()
                                        if key != "document_input"}} if predictor_pair is not None else {}),
            **({"g_quality_target": quality_target} if quality_target is not None else {}),
            **({"exclusion_reason": sorted(set(reasons))} if reasons else {})})
    return rows


__all__ = ["project_result_rows", "derive_g_candidates"]
