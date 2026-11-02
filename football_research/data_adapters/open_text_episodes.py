"""Turn archived acquisitions into response-learning rows, never invented labels.

The adapter retains every actual web return, including failures and index pages.
Only the text returned by this action is a response; history comes from the
public state immediately before it. Result labels and prior-training provenance
are separate supervisor inputs. Missing prerequisites remain explicit exclusions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, parse_qsl

from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical

VERSION = "open-text-episodes-v1"
RUN_VERSIONS = {"learned-voi-open-execution-v1", "learned-voi-open-execution-v2",
                "learned-voi-open-execution-v3", "learned-voi-open-execution-v4"}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _probabilities(value):
    return (isinstance(value, list) and len(value) == 3
            and all(type(x) in {int, float} and math.isfinite(x) and x > 0 for x in value)
            and abs(sum(value) - 1.) < 1e-9)


def document_role(document):
    """A structural audit description, not a relevance or importance label."""
    if document.get("status") != "OK":
        return "failed_return"
    if not document.get("text", "").strip():
        return "empty_return"
    if document.get("kind") in {"search", "discovery"}:
        return "search_results"
    if document.get("kind") == "article":
        return "article_candidate_not_semantically_verified"
    if document.get("kind") == "index":
        return "index_page"
    structured = document.get("structured_news")
    if (isinstance(structured, Mapping)
            and structured.get("version") == "structured-news-discovery-v1"
            and structured.get("role") in {"article", "liveblog"}
            and isinstance(structured.get("segments"), list)):
        body = document.get("text", "")
        for segment in structured["segments"]:
            if not isinstance(segment, Mapping):
                continue
            start, end, text = segment.get("start"), segment.get("end"), segment.get("text")
            if (type(start) is int and type(end) is int and 0 <= start < end <= len(body)
                    and isinstance(text, str) and text.strip() and body[start:end] == text):
                return "article_candidate_not_semantically_verified"
    path = urlsplit(document.get("url", "")).path.rstrip("/").lower()
    if (path in {"", "/en", "/news", "/en/news"}
            or "/fixtures" in path or "/results" in path
            or "fixture-and-event-timings" in path or "/league/" in path):
        return "index_page"
    if re.search(r"/news/[^/]+(?:/[^/]+)?$", path):
        return "article_candidate_not_semantically_verified"
    return "unknown_page_role"


def response_for_model(document):
    """Shared training/runtime projection; tool metadata, never body instructions.

    The trusted tool may explicitly classify an article. Generic ``page``
    remains structurally classified, and unsupported paths remain unknown.
    """
    role = document_role(document)
    return {"status": document.get("status"), "text": document.get("text", ""),
            "kind": {"search_results": "discovery", "index_page": "index",
                     "article_candidate_not_semantically_verified": "article"}.get(role, "unknown")}


def _response(document):
    names = ("document_id", "raw_sha256", "text", "status", "kind", "url",
             "observed_at", "available_at", "source_published_at", "text_truncated")
    result = {name: deepcopy(document.get(name)) for name in names}
    result["document_role"] = document_role(document)
    result["source_kind"] = result["kind"]
    result.update(response_for_model(document))
    result["text_sha256"] = sha256(str(document.get("text", "")).encode()).hexdigest()
    return result


def _label(match_id, kickoff, responses, labels, as_of):
    candidates = labels.get(match_id, [])
    if not candidates:
        return None, ["independent_settlement_missing"]
    eligible = []
    for label in candidates:
        try:
            home, away = label.get("home_goals"), label.get("away_goals")
            clocks = [parse_time(label[k]) for k in ("available_at", "observed_at")]
            if label.get("operational_available_at"):
                clocks.append(parse_time(label["operational_available_at"]))
            if (label.get("status") != "final" or label.get("score_period") != "regulation"
                    or label.get("information_mode") not in {"operational", "historical_reconstruction"}
                    or type(home) is not int or type(away) is not int or min(home, away) < 0
                    or parse_time(label["kickoff"]) != kickoff
                    or min(clocks) <= kickoff or max(clocks) > as_of
                    or not label.get("input_manifest") or not label.get("source_reference")
                    or not label.get("revision_id") or not label.get("source")
                    or label.get("source") in {"model", "synthetic", "unknown"}
                    or any(digest in str(label["input_manifest"]) for digest in responses if digest)):
                continue
            eligible.append(label)
        except (ValueError, TypeError, KeyError):
            continue
    if not eligible:
        return None, ["independent_settlement_unavailable_or_invalid"]
    scores = {(row["home_goals"], row["away_goals"]) for row in eligible}
    if len(scores) != 1:
        return None, ["independent_settlement_conflict"]
    row = min(eligible, key=lambda item: (parse_time(item["available_at"]), item["revision_id"]))
    home, away = row["home_goals"], row["away_goals"]
    return {"outcome": 0 if home > away else 1 if home == away else 2,
            "available_at": max(parse_time(row[k]) for k in
                                ("available_at", "observed_at", "operational_available_at")
                                if row.get(k)).isoformat(),
            "source": row["source"], "source_reference": row["source_reference"],
            "revision_id": row["revision_id"], "input_manifest": deepcopy(row["input_manifest"]),
            "information_mode": row["information_mode"]}, []


def _prior_exclusions(proof, *, match_id, decision_at, model_version,
                      config=None, folder=None, prior_probabilities=None):
    if not isinstance(proof, Mapping):
        return ["forward_prior_training_binding_missing"]
    if proof.get("probability_basis") == "observed_market_snapshot":
        # A deterministic market snapshot has no fitted training provenance.
        # Verify the actual quote/fixture bytes and the resulting p0 instead.
        try:
            from .prospective_text import MarketPriorPolicy, verify_prior_archive, verify_run_identity
            manifest = config["policy_manifest"]
            if config["schema"] in {"learned-voi-open-execution-v3", "learned-voi-open-execution-v4"}:
                manifest = manifest["base_policy"]
            policy = MarketPriorPolicy.from_manifest(manifest)
            verify_prior_archive(manifest, Path(folder)/"prior-archive")
            verify_run_identity(folder, config)
            if (proof != policy.prior_binding() or model_version != policy.model.model_version
                    or match_id != policy.model.fixture["match_id"]
                    or parse_time(policy.model.frozen_at) >= decision_at
                    or prior_probabilities != policy.model.probabilities):
                return ["observed_market_prior_binding_invalid"]
        except (ValueError, TypeError, KeyError, OSError):
            return ["observed_market_prior_archive_invalid"]
        return []
    try:
        ids = proof["training_match_ids"]
        if (proof.get("probability_basis") not in {"forward_oof", "frozen_pre_match"}
                or proof.get("model_version") != model_version
                or not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or match_id in ids or parse_time(proof["training_as_of"]) >= decision_at
                or not isinstance(proof.get("model_sha256"), str)
                or len(proof["model_sha256"]) != 64):
            return ["forward_prior_training_binding_invalid"]
    except (ValueError, TypeError, KeyError):
        return ["forward_prior_training_binding_invalid"]
    return []


def _verify_document(folder, config, receipt):
    """Reproject response bytes with the recorded extractor; no HTTP calls."""
    from football_agent.inquiry_tools import OpenWebTools
    from football_spider.models.contracts import Observation
    from football_spider.storage.raw_archive import RawArchive
    record, document = receipt["receipt"], receipt["document"]
    obs = Observation.model_validate(record["observation"])
    transport = config["supervisor_metadata"]["transport"]
    version = int(transport["version"].rsplit("v", 1)[1])
    archive = RawArchive(folder / "source-archive")
    clock = lambda: parse_time(document["available_at"])
    if "news_discovery" in transport:
        from football_agent.news_discovery import NewsDiscoveryTools
        tools = NewsDiscoveryTools.from_configuration(archive, transport, clock=clock)
    else:
        if transport.get("search_provider", "bing_rss") != "bing_rss":
            raise ValueError("unsupported_recorded_search_provider")
        tools = OpenWebTools(archive, transport["allowed_hosts"], extraction_version=version, clock=clock)
    projected = tools.project(obs, record["prepared"])
    # v1/v2 query spelling predates ordered URL canonicalization.
    def identity(url):
        parts = urlsplit(url)
        return parts.scheme, parts.netloc, parts.path, sorted(parse_qsl(parts.query)), parts.fragment
    if version < 3 and identity(projected["url"]) == identity(document["url"]):
        projected["url"] = document["url"]
    if projected != document:
        raise ValueError("document_does_not_reproduce_raw_archive")
    return document


def build_open_text_episodes(run_folders: Iterable[str | Path], *, settlements=(),
                            prior_bindings=None, as_of):
    """Return ``{episodes, audit}`` for archived web acquisitions.

    ``settlements`` uses data_adapters.settlement.export_results rows. The caller
    supplies independently verified normalized result evidence; this function
    checks eligibility rather than downloading or guessing scores.
    ``prior_bindings`` maps model versions to the saved training ID manifest,
    model hash, training_as_of and probability_basis (forward_oof/frozen_pre_match).
    These supervisor attestations are reported separately from raw-byte checks.
    Unknown action propensities stay null and cannot support off-policy claims.
    """
    as_of = parse_time(as_of)
    labels = defaultdict(list)
    for row in settlements:
        labels[row.get("match_id")].append(row)
    prior_bindings = prior_bindings or {}
    episodes, runs, rejected, duplicates, seen = [], [], [], [], {}
    for folder_value in sorted(set(str(Path(p).resolve()) for p in run_folders)):
        folder = Path(folder_value)
        try:
            run, config = _read(folder / "run.json"), _read(folder / "configuration.json")
            if run.get("schema") not in RUN_VERSIONS:
                raise ValueError("unsupported_run_schema")
            if run["configuration_digest"] != _digest(config):
                raise ValueError("configuration_digest_mismatch")
            run_id, run_hash = run["run_id"], _digest(run)
            if run_id in seen:
                if seen[run_id] != run_hash:
                    raise ValueError("duplicate_run_id_has_conflicting_content")
                duplicates.append(str(folder))
                continue
            # Prefer original directories with the byte archive over replay copies.
            if not (folder / "source-archive" / "raw").is_dir():
                rejected.append({"path": str(folder), "reason": "original_response_archive_missing"})
                continue
            state = config["initial_state"]
            match_id = state["match_id"]
            kickoff, cutoff = parse_time(state["base"]["kickoff"]), parse_time(state["prediction_cutoff"])
            if cutoff >= kickoff:
                raise ValueError("prematch_cutoff_not_before_kickoff")
            seen[run_id] = run_hash
            runs.append({"path": str(folder), "run_id": run_id, "run_sha256": run_hash,
                         "match_id": match_id, "kickoff_at": kickoff.isoformat()})
            past = {}
            for cycle in run["cycles"]:
                action = cycle["selected_action"]
                if action is None:
                    continue
                if action.get("tool") not in {"read_url", "search_web"}:
                    continue
                step = cycle["step"]
                reasons = []
                try:
                    if _read(folder / f"cycle-{step:03}.json") != cycle:
                        raise ValueError("cycle_snapshot_mismatch")
                    public = cycle["public_before"]
                    if _digest(public) != cycle["public_digest"]:
                        raise ValueError("public_state_digest_mismatch")
                    if action not in cycle["generated_actions"]:
                        raise ValueError("selected_action_not_in_generated_menu")
                    returned = _read(folder / f"receipt-{step:03}.json")
                    document = _verify_document(folder, config, returned)
                    decision = parse_time(public.get("decision_at") or
                                          returned["receipt"]["observation"]["context"]["requested_at"])
                    observed, available = parse_time(document["observed_at"]), parse_time(document["available_at"])
                    if not decision <= observed <= available <= min(cutoff, as_of):
                        reasons.append("response_not_operationally_available_before_cutoff")
                    returned_at = cycle.get("returned_at")
                    if run["schema"] in {"learned-voi-open-execution-v3", "learned-voi-open-execution-v4"} and returned_at is None:
                        reasons.append("actual_return_clock_missing")
                    elif returned_at is not None:
                        returned_at = parse_time(returned_at)
                        if returned_at < decision or available > returned_at:
                            reasons.append("response_not_available_at_actual_return")
                    admission = cycle.get("admission")
                    # An archived body can be available before kickoff yet
                    # still have been rejected by the actor at this attempt.
                    # Failed/NO_DATA returns are valid neutral observations;
                    # only successful bodies require actor admission.
                    provider_status = returned.get("status")
                    if provider_status not in {"OK", "FAILED", "NO_DATA"}:
                        reasons.append("provider_response_status_invalid")
                    if provider_status == "OK" and (
                            run["schema"] in {"learned-voi-open-execution-v3", "learned-voi-open-execution-v4"} or admission is not None):
                        if not isinstance(admission, Mapping) or admission.get("status") != "admitted":
                            reasons.append("successful_response_not_admitted")
                    if cycle.get("execution_status", "returned") != "returned":
                        reasons.append("response_execution_not_reconciled")
                    response = _response(document)
                    if provider_status in {"FAILED", "NO_DATA"}:
                        # The executor maps malformed/empty provider content
                        # to its recorded neutral outcome. Match that exact
                        # runtime outcome instead of teaching INVALID_CONTENT
                        # as if it were a supported textual response.
                        response.update(status=provider_status, kind="unknown")
                    before, p0 = cycle["prediction_before"], run["initial_probabilities"]
                    if (not _probabilities(before) or not _probabilities(p0)
                            or public["prediction"].get("probabilities") != before):
                        reasons.append("public_prior_probabilities_invalid")
                    # Selection support is conditional on this actual generated menu.
                    explore = cycle.get("exploration_record")
                    propensity = None
                    if explore:
                        probability = explore.get("selected_p")
                        probabilities = explore.get("probabilities", {})
                        if (type(probability) not in {int, float} or not 0 < probability <= 1
                                or explore.get("chosen_id") != action["action_id"]
                                or probabilities.get(action["action_id"]) != probability
                                or abs(sum(probabilities.values()) - 1.) > 1e-9):
                            reasons.append("logged_selection_probability_invalid")
                        else:
                            propensity = probability
                    visible = {d["document_id"]: d for d in public.get("documents", [])}
                    history, history_training_reasons = [], []
                    for attempt in public.get("attempts", []):
                        if attempt["action_id"] not in past:
                            continue  # Structured actions are not text observations.
                        previous = past[attempt["action_id"]]
                        shown = visible.get(previous["response"]["document_id"])
                        if shown:
                            if parse_time(shown["available_at"]) > decision:
                                reasons.append("future_document_in_public_history")
                            if _response(shown) != previous["response"]:
                                reasons.append("history_document_differs_from_actual_return")
                        historical_response = _response(shown) if shown else {
                            "status": attempt["status"], "text": "",
                            "kind": "unknown", "document_id": None,
                            "document_role": "not_admitted"}
                        if (historical_response["status"] == "OK"
                                and historical_response["kind"] == "unknown"):
                            history_training_reasons.append("history_response_kind_unknown")
                        if not previous["response_observation_eligible"]:
                            history_training_reasons.append("history_response_not_operationally_eligible")
                        history.append({"action": deepcopy(previous["action"]),
                                        "response": historical_response})
                    label, label_reasons = _label(match_id, kickoff, [document["raw_sha256"]], labels, as_of)
                    version = public["prediction"].get("initial_model_version", public["prediction"].get("model_version"))
                    proof = prior_bindings.get(version)
                    training_reasons = reasons + history_training_reasons + label_reasons + _prior_exclusions(
                        proof, match_id=match_id, decision_at=decision, model_version=version,
                        config=config, folder=folder, prior_probabilities=p0)
                    if state.get("information_mode") != "operational":
                        training_reasons.append("prior_information_mode_not_operational")
                    if provider_status == "OK" and response["kind"] == "unknown":
                        training_reasons.append("response_kind_unknown")
                    item = {"schema": VERSION, "episode_id": f"{run_id}:{step}", "run_id": run_id,
                            "match_id": match_id, "step": step, "kickoff_at": kickoff.isoformat(),
                            "prediction_cutoff": cutoff.isoformat(), "decision_at": decision.isoformat(),
                            "observed_at": observed.isoformat(), "available_at": available.isoformat(),
                            "prior_probabilities": deepcopy(p0), "decision_probabilities": deepcopy(before),
                            "prior_model_version": version,
                            "action": deepcopy(action), "response": response, "history": history,
                            "generated_candidate_ids": [a["action_id"] for a in cycle["generated_actions"]],
                            "selection_probability": propensity,
                            "selection_scope": "conditional_on_generated_eligible_menu_only",
                            "selection_basis": cycle.get("selection_basis"),
                            "label": label, "prior_training_binding": deepcopy(proof),
                            "response_observation_eligible": not reasons,
                            "numeric_voi_training_eligible": not training_reasons,
                            "off_policy_eligible": not training_reasons and propensity is not None,
                            "training_exclusion_reasons": sorted(set(training_reasons)),
                            "lineage": {"run_path": str(folder), "run_sha256": run_hash,
                                        "public_before_sha256": cycle["public_digest"],
                                        "receipt_sha256": _digest(returned),
                                        "raw_response_reprojected": True,
                                        "supervisor_bindings_independently_audited_by_adapter": False}}
                    episodes.append(item)
                    past[action["action_id"]] = item
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    rejected.append({"path": str(folder), "step": step, "reason": str(exc)})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            rejected.append({"path": str(folder), "reason": str(exc)})
    audit = {"schema": VERSION, "as_of": as_of.isoformat(), "runs": runs,
             "unique_runs": len(runs), "unique_matches": len({r["match_id"] for r in runs}),
             "episodes": len(episodes), "rejected": rejected, "duplicate_run_directories": duplicates,
             "response_roles": dict(Counter(e["response"]["document_role"] for e in episodes)),
             "response_observation_eligible": sum(e["response_observation_eligible"] for e in episodes),
             "independent_labels_attached": sum(e["label"] is not None for e in episodes),
             "numeric_voi_training_eligible": sum(e["numeric_voi_training_eligible"] for e in episodes),
             "exclusion_counts": dict(Counter(r for e in episodes for r in e["training_exclusion_reasons"])),
             "episodes_sha256": _digest(episodes), "network_requests": 0,
             "scope": "logged_action_responses_not_complete_counterfactual_returns_or_open_world_support"}
    return {"episodes": episodes, "audit": audit}


def training_records(dataset):
    """Project qualified episodes to the text response model's supervised API.

    A successful empty selection is a valid return when no real labels exist.
    Sidecars remain in the complete dataset. No label or admission is imputed.
    """
    result = []
    for episode in dataset["episodes"]:
        if (not episode["numeric_voi_training_eligible"]
                or episode["training_exclusion_reasons"] or episode["label"] is None):
            continue
        result.append({key: deepcopy(episode[key]) for key in (
            "match_id", "prediction_cutoff", "prior_probabilities", "action", "response", "history")}
            | {"outcome": episode["label"]["outcome"],
               "result_available_at": episode["label"]["available_at"]})
    return result


def empirical_risk_records(dataset):
    """Recover actual per-step clocks for the fixed-predictor risk bridge.

    Input must be a complete dataset produced by the existing raw/settlement
    gateway. Its label provenance remains a trusted gateway contract; this
    projection cannot establish independent settlement from an outcome integer.
    The legacy training_records projection is deliberately unchanged.
    """
    episodes, source_audit = dataset["episodes"], dataset["audit"]
    if source_audit.get("schema") != VERSION or source_audit.get("episodes_sha256") != _digest(episodes):
        raise ValueError("Risk records require the complete unchanged gateway episode dataset")
    index = defaultdict(list)
    for episode in episodes:
        index[(episode["run_id"], episode["action"]["action_id"])].append(episode)

    def eligible(row):
        return (row.get("numeric_voi_training_eligible") is True and row.get("response_observation_eligible") is True
                and row.get("training_exclusion_reasons") == [] and isinstance(row.get("label"), Mapping))

    def signature(step):
        return {"action": step["action"], "response": {key: step["response"].get(key)
                for key in ("status", "kind", "text")}}

    def clocks(row):
        names = ("decision_at", "observed_at", "available_at")
        if any(not row.get(key) for key in names):
            raise ValueError("actual_step_clock_missing")
        parsed = [parse_time(row[key]) for key in names]
        if not parsed[0] <= parsed[1] <= parsed[2] <= parse_time(row["prediction_cutoff"]):
            raise ValueError("actual_step_clock_order_or_cutoff_invalid")
        for key, stamp in zip(names[1:], parsed[1:]):
            if row["response"].get(key) is not None and parse_time(row["response"][key]) != stamp:
                raise ValueError("response_clock_differs_from_executed_episode")
        return {key: stamp.isoformat() for key, stamp in zip(names, parsed)}

    records, exclusions = [], []
    for episode in episodes:
        try:
            if not eligible(episode):
                raise ValueError("gateway_episode_not_training_eligible")
            label = episode["label"]
            if (type(label.get("outcome")) is not int or label["outcome"] not in (0, 1, 2)
                    or any(not label.get(key) for key in ("source", "source_reference", "revision_id", "input_manifest"))):
                raise ValueError("independent_gateway_label_provenance_missing")
            cutoff, kickoff, settled = (parse_time(episode["prediction_cutoff"]),
                                        parse_time(episode["kickoff_at"]), parse_time(label["available_at"]))
            if not cutoff < kickoff < settled <= parse_time(source_audit["as_of"]):
                raise ValueError("gateway_label_chronology_invalid")
            current, history, previous_step, previous_available = clocks(episode), [], -1, None
            for position, step in enumerate(episode["history"]):
                matches = index.get((episode["run_id"], step["action"]["action_id"]), [])
                if len(matches) != 1:
                    raise ValueError("history_episode_missing_or_ambiguous_in_same_run")
                prior = matches[0]
                if not eligible(prior):
                    raise ValueError("history_episode_not_training_eligible")
                if (not previous_step < prior["step"] < episode["step"]
                        or any(prior[key] != episode[key] for key in
                               ("match_id", "prior_probabilities", "prediction_cutoff", "kickoff_at", "label"))):
                    raise ValueError("history_order_or_match_context_conflict")
                if (signature(prior) != signature(step) or [signature(p) for p in prior["history"]]
                        != [signature(p) for p in episode["history"][:position]]):
                    raise ValueError("history_differs_from_actual_executed_prefix")
                observed = clocks(prior)
                available, decision = parse_time(observed["available_at"]), parse_time(observed["decision_at"])
                if available > parse_time(current["decision_at"]) or previous_available is not None and previous_available > decision:
                    raise ValueError("history_response_unavailable_at_next_decision")
                history.append({"action": deepcopy(step["action"]), "response": deepcopy(step["response"]), **observed})
                previous_step, previous_available = prior["step"], available
            records.append({key: deepcopy(episode[key]) for key in
                            ("match_id", "prediction_cutoff", "prior_probabilities", "action", "response")}
                | {"history": history, "kickoff": episode["kickoff_at"], "outcome": label["outcome"],
                   "result_available_at": label["available_at"], **current})
        except (ValueError, KeyError, TypeError) as error:
            exclusions.append({"episode_id": episode.get("episode_id"), "run_id": episode.get("run_id"),
                "step": episode.get("step"), "reason": str(error),
                "gateway_exclusion_reasons": deepcopy(episode.get("training_exclusion_reasons", []))})
    return {"records": records, "audit": {"schema": "empirical-text-risk-records-v1",
        "input_episodes": len(episodes), "output_records": len(records),
        "unique_matches": len({row["match_id"] for row in records}), "exclusions": exclusions,
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "source_episodes_sha256": source_audit["episodes_sha256"], "records_sha256": _digest(records),
        "clock_scope": "same_run_executed_episode_clocks_never_imputed",
        "label_scope": "trust_existing_raw_settlement_gateway_not_integer_only_verification",
        "network_requests": 0, "production_ready": False}}
