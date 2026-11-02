"""Finite, researcher-acquired response snapshots; never actor counterfactual truth.

Only public_state/reveal form the actor interface. This is data isolation, not
a sandbox against Python code inspecting private evaluator objects or files.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from football_agent.inquiry_tools import OpenWebTools
from football_agent.match_evidence import _identity, _mentions, _fixture_spans
from football_research.acquisition.state import parse_time
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .open_text_episodes import _digest, _read, _verify_document, document_role, RUN_VERSIONS
from .prospective_text import verify_run_identity

VERSION = "finite-information-panel-v1"
DERIVED_VERSION = "finite-information-panel-v2"
SCOPE = "finite_frozen_menu_snapshot_not_open_world_or_full_policy_counterfactual"


def _write(path, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _tools(archive, config, clock):
    if "news_discovery" in config:
        from football_agent.news_discovery import NewsDiscoveryTools
        return NewsDiscoveryTools.from_configuration(archive, config, clock=clock)
    if config.get("search_provider", "bing_rss") != "bing_rss":
        raise ValueError("Unknown panel extraction transport")
    return OpenWebTools(archive, config["allowed_hosts"], clock=clock,
                        extraction_version=int(config["version"].rsplit("v", 1)[1]))


def _snapshot(folder, step):
    folder = Path(folder).resolve()
    run, config = _read(folder / "run.json"), _read(folder / "configuration.json")
    if run.get("schema") not in RUN_VERSIONS or run["configuration_digest"] != _digest(config):
        raise ValueError("Actor run configuration binding differs")
    if not run.get("finished_at") or not run.get("terminal_selection"):
        raise ValueError("Panel requires a completed actor run")
    cycles = [c for c in run["cycles"] if c["step"] == step]
    if len(cycles) != 1:
        raise ValueError("Frozen actor cycle missing or ambiguous")
    cycle = cycles[0]
    terminal_snapshot = (cycle.get("selected_action") is None and cycle.get("execution_status") == "not_requested"
        and cycle.get("terminal") == run["terminal_selection"] and cycle == run["cycles"][-1])
    if (_read(folder / f"cycle-{step:03}.json") != cycle
            or (cycle.get("public_digest") != _digest(cycle["public_before"])
                and not ("public_digest" not in cycle and terminal_snapshot))):
        raise ValueError("Actor cycle or public snapshot binding differs")
    identity = verify_run_identity(folder, config)
    if identity is None:
        raise ValueError("Panel requires the raw-verified fixture identity")
    public = deepcopy(cycle["public_before"])
    for document in public["documents"]:
        receipts = [_read(folder / f"receipt-{c['step']:03}.json") for c in run["cycles"]
                    if c["step"] < step and c.get("selected_action")]
        matching = [r for r in receipts if r.get("document", {}).get("document_id") == document["document_id"]]
        if (len(matching) != 1 or _verify_document(folder, config, matching[0]) != document
                or parse_time(document["available_at"]) > parse_time(public["decision_at"])):
            raise ValueError("Visible actor history is not its actual prior acquired document")
    public.setdefault("task", {})["seed_urls"] = deepcopy(config["supervisor_metadata"]["generator"]["seed_urls"])
    match = deepcopy(config["evidence_projection"]["match"])
    match.update(identity_binding_sha256=identity["binding_sha256"],
                 source_fixture_url=identity["canonical"]["source_url"])
    if (match["match_id"] != config["initial_state"]["match_id"] or
            parse_time(match["prediction_cutoff"]) != parse_time(config["initial_state"]["prediction_cutoff"])):
        raise ValueError("Frozen match and actor state disagree")
    bindings = {name: sha256((folder / name).read_bytes()).hexdigest()
                for name in ("run.json", "configuration.json", f"cycle-{step:03}.json")}
    return run, cycle, public, match, config["supervisor_metadata"]["transport"], bindings


def freeze_panel_design(run_folder, *, step, candidate_ids=None, match_scope=None, frozen_at, output,
                        discovery_version=None):
    """Freeze the first two feasible novel reads, before looking at their bodies."""
    run, cycle, public, match, transport, bindings = _snapshot(run_folder, step)
    frozen = parse_time(frozen_at)
    if not parse_time(run["finished_at"]) <= frozen < parse_time(match["prediction_cutoff"]):
        raise ValueError("Panel design must follow the completed run and precede prediction cutoff")
    if match_scope is not None and match_scope != match:
        raise ValueError("Match scope must equal the verified actor binding")
    menu = cycle["generated_actions"]
    derived = None
    if discovery_version is not None:
        # A new menu is research material observed NOW, not a rewritten actor
        # decision. Only an explicit supported successor may derive from the
        # original generator; no caller-provided URLs, topics or costs.
        from football_agent.evidence_candidates import EvidenceDocumentCandidates
        if type(discovery_version) is not int or discovery_version != 8:
            raise ValueError("Derived panels support explicit discovery version 8 only")
        original = _read(Path(run_folder)/"configuration.json")["supervisor_metadata"]["generator"]
        generator = EvidenceDocumentCandidates.from_configuration(original)
        if generator(cycle["public_before"]) != menu:
            raise ValueError("Original menu does not reproduce original actor state")
        if generator._discovery_version >= discovery_version:
            raise ValueError("A derived panel requires a newer generator than the original")
        keys = ("match_terms", "seed_urls", "allowed_hosts", "numeric_actions", "web_cost",
                "match", "profile", "max_discovered_links", "max_followup_queries", "max_candidates")
        successor = EvidenceDocumentCandidates(**{k: deepcopy(original[k]) for k in keys},
                                               discovery_version=discovery_version)
        public["decision_at"] = frozen.isoformat()
        menu = successor(public)
        public["candidate_contexts"] = successor.candidate_contexts(public, menu)
        derived = {"discovery_version": discovery_version, "generator": successor.configuration(),
            "generated_menu": menu, "available_at": frozen.isoformat(),
            "source": "original_observed_parent_links_only", "original_actor_modified": False,
            "actor_continuation": False, "research_requests_charged_separately": True,
            "automatic_training_admission": False}
    tools = _tools(RawArchive(Path(run_folder) / "source-archive"), transport, lambda: frozen)
    attempted = {c["selected_action"]["action_id"] for c in run["cycles"] if c.get("selected_action")}
    requested = {_digest(_read(Path(run_folder) / f"receipt-{c['step']:03}.json")["receipt"]["prepared"])
                 for c in run["cycles"] if (c.get("selected_action") or {}).get("tool") in {"read_url", "search_web"}}
    candidates, prepared, seen, exclusions = [], {}, set(), []
    for action in menu:
        try:
            if action.get("tool") != "read_url" or not action.get("available", True) or action["action_id"] in attempted:
                raise ValueError("not_a_novel_available_read")
            request = tools.prepare(action, public)
            key = _digest(request)
            if key in requested or key in seen:
                raise ValueError("already_acquired_or_duplicate_request")
            seen.add(key)
            candidates.append(deepcopy(action))
            prepared[action["action_id"]] = request
        except (ValueError, KeyError) as error:
            exclusions.append({"action_id": action.get("action_id"), "reason": str(error)})
    selected = candidates[:2]
    if candidate_ids is not None and list(candidate_ids) != [a["action_id"] for a in selected]:
        raise ValueError("Panel selection must be the first two feasible novel reads in frozen menu order")
    design = {"schema": VERSION, "scope": SCOPE, "actor_run_folder": str(Path(run_folder).resolve()),
        "actor_run_id": run["run_id"], "actor_step": step, "actor_bindings": bindings,
        "frozen_at": frozen.isoformat(), "match_scope": match, "transport": transport,
        "actor_public_before": cycle["public_before"], "actor_public_before_sha256": _digest(cycle["public_before"]),
        "public_digest_basis": "runtime_recorded" if "public_digest" in cycle else "derived_from_bound_terminal_snapshot",
        "public_before": public, "initial_probabilities": run["initial_probabilities"],
        "original_generated_menu": cycle["generated_actions"], "candidates": selected,
        "prepared": {a["action_id"]: prepared[a["action_id"]] for a in selected},
        "candidate_exclusions": exclusions, "selection_rule": "first_two_feasible_novel_reads_in_actual_menu_order"}
    if derived is not None:
        design.update(schema=DERIVED_VERSION, derived_menu=derived,
            selection_rule="first_two_feasible_novel_reads_in_rederived_observed_menu_order")
    design["panel_id"] = "panel-" + _digest(design)[:24]
    if output is not None:
        _write(output, design)
    return design


def _check_design(design):
    if (design.get("schema") not in {VERSION, DERIVED_VERSION} or design.get("panel_id") !=
            "panel-" + _digest({k: v for k, v in design.items() if k != "panel_id"})[:24]):
        raise ValueError("Changed panel design")
    rebuilt = freeze_panel_design(design["actor_run_folder"], step=design["actor_step"],
        candidate_ids=[a["action_id"] for a in design["candidates"]], match_scope=design["match_scope"],
        frozen_at=design["frozen_at"], output=None,
        discovery_version=(design["derived_menu"]["discovery_version"] if design["schema"] == DERIVED_VERSION else None))
    if rebuilt != design:
        raise ValueError("Panel no longer matches the frozen actor archive")


def _qualification(document, match):
    role = document_role(document)
    identity = _identity(match)
    mentioned = _mentions(document.get("text", ""), identity)
    spans = _fixture_spans(document, identity) if mentioned else []
    bound = bool(mentioned) and (document["url"] == match["source_fixture_url"]
        or any(s["status"] == "explicit_pair_and_date" for s in spans))
    return {"document_role": role, "target_scope": "match_specific" if bound else "team_context" if mentioned else "unscoped",
        "mentioned_sides": mentioned, "fixture_assertions": spans,
        "applicability_status": "source_assertion_not_independent_fact_or_value_verification",
        "article": role == "article_candidate_not_semantically_verified" and bool(document.get("text", "").strip()),
        "text_sha256": sha256(document.get("text", "").encode()).hexdigest()}


def _check_captures(design, captures, archive_folder, panel_as_of):
    as_of, frozen = parse_time(panel_as_of), parse_time(design["frozen_at"])
    if not frozen <= as_of < parse_time(design["match_scope"]["prediction_cutoff"]):
        raise ValueError("Panel snapshot precedes design or is at/after prediction cutoff")
    archive = RawArchive(Path(archive_folder))
    saved = {o.id: o for o in archive.observations()}
    actions = {a["action_id"]: a for a in design["candidates"]}
    checked, seen, article_hashes, raw_hashes = [], set(), set(), set()
    visible_texts = {sha256(d.get("text", "").encode()).hexdigest() for d in design["public_before"]["documents"]}
    visible_raw = {d.get("raw_sha256") for d in design["public_before"]["documents"]}
    for capture in captures:
        action_id, result = capture["action_id"], capture["result"]
        if (set(capture) not in ({"action_id", "decision_at", "returned_at", "prepared", "result"},
                {"action_id", "decision_at", "returned_at", "prepared", "result", "qualification"})
                or set(result) != {"document", "observation", "request_id", "http_attempts", "latency_seconds", "money", "money_status"}):
            raise ValueError("Unexpected capture fields; labels require a separate sidecar")
        if action_id not in actions or action_id in seen:
            raise ValueError("Capture is outside frozen panel or duplicated")
        seen.add(action_id)
        obs = Observation.model_validate(result["observation"])
        document, request = result["document"], capture["prepared"]
        clocks = [frozen, parse_time(capture["decision_at"]), parse_time(obs.context["requested_at"]),
                  obs.observed_at, parse_time(document["available_at"]), parse_time(capture["returned_at"]), as_of]
        if clocks != sorted(clocks):
            raise ValueError("Actual panel request/response clocks cannot be backdated")
        if (obs.id not in saved or saved[obs.id] != obs or obs.context.get("inquiry_run_id") != design["panel_id"]
                or obs.context.get("inquiry_request_id") != result["request_id"]
                or request != design["prepared"][action_id]
                or (obs.source, obs.endpoint, obs.params) != (request["source"], request["endpoint"], request["params"])):
            raise ValueError("Panel response is not bound to its actual archived request")
        tools = _tools(archive, design["transport"], lambda: parse_time(document["available_at"]))
        if tools.prepare(actions[action_id], design["public_before"]) != request or tools.project(obs, request) != document:
            raise ValueError("Panel response does not reproduce original archived bytes")
        latency, money = result["latency_seconds"], result["money"]
        if (result["http_attempts"] != 1 or type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0
                or money is not None and (type(money) not in {int, float} or not math.isfinite(money) or money < 0)):
            raise ValueError("Invalid actual panel resource receipt")
        qualification = _qualification(document, design["match_scope"])
        qualification["already_visible_content"] = qualification["text_sha256"] in visible_texts or obs.content_hash in visible_raw
        qualification["duplicate_article_content"] = qualification["article"] and (
            qualification["text_sha256"] in article_hashes or obs.content_hash in raw_hashes)
        if qualification["article"]:
            article_hashes.add(qualification["text_sha256"])
            raw_hashes.add(obs.content_hash)
        checked.append({**deepcopy(capture), "qualification": qualification})
    actual_ids = {o.id for o in saved.values() if o.context.get("inquiry_run_id") == design["panel_id"]}
    if actual_ids != {row["result"]["observation"]["id"] for row in captures}:
        raise ValueError("Panel captures omit actual archived attempts; retries cannot be hidden")
    article_count = sum(r["qualification"]["article"] and not r["qualification"]["duplicate_article_content"]
                        and not r["qualification"]["already_visible_content"] for r in checked)
    audit = {"expected_candidates": len(actions), "actual_returns": len(checked),
        "actual_http_attempts": len(actual_ids),
        "missing_action_ids": sorted(set(actions) - seen), "fully_observed_panel": len(actions) == len(checked) == 2,
        "distinct_article_contents": article_count, "multi_article_panel": article_count >= 2,
        "article_independence": "nonidentical_bytes_and_text_do_not_establish_independent_facts",
        "scope": SCOPE, "label_sidecar": "absent_separate_independent_settlement_required",
        "network_requests_during_validation": 0}
    return checked, audit


def seal_panel(design_path, captures, *, panel_as_of, archive_folder, output):
    """Save actual researcher returns; partial/failure panels remain explicit."""
    design = _read(design_path)
    _check_design(design)
    checked, audit = _check_captures(design, captures, archive_folder, panel_as_of)
    panel = {"schema": design["schema"], "design": design, "design_sha256": _digest(design),
        "archive_folder": str(Path(archive_folder).resolve()), "panel_as_of": parse_time(panel_as_of).isoformat(),
        "captures": checked, "audit": audit}
    panel["panel_sha256"] = _digest(panel)
    _write(output, panel)
    return panel


def load_panel(path):
    """Zero-network validation; historical load time never changes snapshot time."""
    panel = _read(path)
    if (set(panel) != {"schema", "design", "design_sha256", "archive_folder", "panel_as_of", "captures", "audit", "panel_sha256"}
            or panel["schema"] not in {VERSION, DERIVED_VERSION} or panel["schema"] != panel["design"]["schema"]
            or panel["panel_sha256"] != _digest({k: v for k, v in panel.items() if k != "panel_sha256"})
            or panel["design_sha256"] != _digest(panel["design"])):
        raise ValueError("Changed panel manifest or unexpected label fields")
    _check_design(panel["design"])
    checked, audit = _check_captures(panel["design"], panel["captures"], panel["archive_folder"], panel["panel_as_of"])
    if checked != panel["captures"] or audit != panel["audit"]:
        raise ValueError("Panel qualification does not reproduce original responses")
    return InformationPanel(panel)


class InformationPanel:
    def __init__(self, validated_panel):
        self._panel, self._revealed = deepcopy(validated_panel), []

    def public_state(self):
        design = self._panel["design"]
        public = deepcopy(design["public_before"] if design["schema"] == DERIVED_VERSION else design["actor_public_before"])
        public.update(panel_id=design["panel_id"], panel_as_of=self._panel["panel_as_of"],
            decision_at=self._panel["panel_as_of"], candidate_menu=deepcopy(design["candidates"]),
            initial_probabilities=deepcopy(design["initial_probabilities"]), panel_scope=SCOPE,
            panel_budget={"requests_limit": len(design["candidates"]), "requests_spent": len(self._revealed)})
        for capture in self._revealed:
            document = capture["result"]["document"]
            if design["schema"] == DERIVED_VERSION:
                action = next(a for a in design["candidates"] if a["action_id"] == capture["action_id"])
                if document["status"] == "OK":
                    public["documents"].append(deepcopy(document))
                public["attempts"].append({"action_id": capture["action_id"], "status": document["status"],
                    "information_key": action["information_key"],
                    "admission": "admitted" if document["status"] == "OK" else "not_admitted"})
                public["state"]["action_history"].append({"action_id": capture["action_id"]})
            else:
                public["documents"].append(deepcopy(document))
                public["attempts"].append({"action_id": capture["action_id"], "status": document["status"]})
        if design["schema"] == DERIVED_VERSION:
            from football_agent.candidate_context import build_candidate_contexts
            public["historical_actor_budget"] = public.pop("budget", None)
            credits = {a["action_id"]: a["cost"]["money"] for a in design["candidates"]}
            public["budget"] = {"limits": {"requests": len(design["candidates"]), "money": sum(credits.values())},
                "spent": {"requests": len(self._revealed),
                          "money": sum(credits[c["action_id"]] for c in self._revealed)}}
            public["candidate_contexts"] = build_candidate_contexts(public, design["candidates"],
                allowed_hosts=design["transport"]["allowed_hosts"])
        return public

    def reveal(self, action_id):
        if action_id in {row["action_id"] for row in self._revealed}:
            raise ValueError("Panel action already revealed")
        captures = [row for row in self._panel["captures"] if row["action_id"] == action_id]
        if len(captures) != 1:
            raise ValueError("No actual captured response; unknown is not an empty return")
        capture = deepcopy(captures[0])
        if parse_time(capture["result"]["document"]["available_at"]) > parse_time(self._panel["panel_as_of"]):
            raise ValueError("Response unavailable at frozen panel snapshot")
        visible = self._panel["design"]["actor_public_before"]["documents"] + [r["result"]["document"] for r in self._revealed]
        document = capture["result"]["document"]
        # The researcher's all-capture duplicate flag is private: exposing it
        # could reveal equality with another, still-unread candidate's body.
        capture["qualification"]["duplicate_article_content"] = any(
            d.get("text") == document.get("text") or d.get("raw_sha256") == document.get("raw_sha256") for d in visible)
        self._revealed.append(capture)
        return deepcopy({"action_id": action_id, "document": capture["result"]["document"],
            "declared_cost": next(a.get("cost") for a in self._panel["design"]["candidates"] if a["action_id"] == action_id),
            "actual_resources": {k: capture["result"][k] for k in ("http_attempts", "latency_seconds", "money", "money_status")},
            "qualification": capture["qualification"], "network_requests": 0, "scope": SCOPE})
