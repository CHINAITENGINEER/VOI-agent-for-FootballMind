"""Raw-backed, match-grouped visible prefixes for the content predictor.

Only the original saved actor states are encoded.  A later parser, research
panel, settlement, or unexecuted candidate never supplies actor-visible text.
Unknown response roles are masks, not the old return model's eligibility gate.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.state import parse_time
from football_research.acquisition.visible_state import encode_visible_state
from .open_text_episodes import _label
from .prospective_text import join_settlements, _digest
from .prospective_training import _registered_runs

VERSION = "raw-backed-context-prefixes-v1"
CONTEXT_RUNTIME_VERSION = "learned-voi-open-execution-v4"
_MODEL_ONLY_REASONS = {"response_kind_unknown", "history_response_kind_unknown"}
_LABEL_REASONS = {"independent_settlement_missing", "independent_settlement_unavailable_or_invalid",
                  "independent_settlement_conflict"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def _bindings(paths):
    files = set()
    for path in map(Path, paths):
        files.update([path] if path.is_file() else (item for item in path.rglob("*") if item.is_file()))
    return {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def _visible_prefixes(run, episodes, identity, labels, as_of):
    """Encode a fully verified run, including its actual terminal snapshot."""
    if run.get("schema") == CONTEXT_RUNTIME_VERSION:
        return _immediate_context_prefixes(run, episodes, identity, labels, as_of)
    cycles = run["cycles"]
    if not cycles or cycles[-1].get("selected_action") is not None:
        raise ValueError("A complete actual terminal public snapshot is required")
    if any(c.get("selected_action") is None for c in cycles[:-1]):
        raise ValueError("An actual run cannot continue after its terminal snapshot")
    selected = [c["step"] for c in cycles[:-1]]
    if (selected != sorted(selected) or len(set(selected)) != len(selected)
            or sorted(e["step"] for e in episodes) != selected):
        raise ValueError("Complete actual response history is required, without dropped steps")
    by_step = {e["step"]: e for e in episodes}
    operational_errors = sorted({reason for e in episodes for reason in e["training_exclusion_reasons"]
                                if reason not in _MODEL_ONLY_REASONS | _LABEL_REASONS})
    if operational_errors or any(not e["response_observation_eligible"] for e in episodes):
        raise ValueError("Invalid operational prefix: "+", ".join(operational_errors))
    public0 = cycles[0]["public_before"]
    mid = identity["original_match_id"]
    cutoff = parse_time(public0["state"]["prediction_cutoff"])
    kickoff = parse_time(public0["state"]["base"]["kickoff"])
    label, label_reasons = _label(mid, kickoff, [e["response"]["raw_sha256"] for e in episodes], labels, as_of)
    prefixes, transitions = [], []
    for index, cycle in enumerate(cycles):
        public = cycle["public_before"]
        visible = encode_visible_state(public, cycle["generated_actions"])
        before = visible["before"]
        if (before["match_id"] != mid or parse_time(before["prediction_cutoff"]) != cutoff
                or parse_time(before["fixture"]["kickoff"]) != kickoff
                or not parse_time(before["decision_at"]) <= min(cutoff, as_of) or cutoff >= kickoff):
            raise ValueError("Visible prefix identity or operational clocks differ")
        if not before["history"]["order_matches_attempts"]:
            raise ValueError("Visible action order and actual attempts disagree")
        prefix_id = f"{run['run_id']}:prefix:{index}"
        row = {"schema": VERSION, "prefix_id": prefix_id, "match_id": mid,
            "physical_match_id": identity["source_event_id"], "run_id": run["run_id"],
            "prefix_index": index, "kickoff_at": kickoff.isoformat(),
            "prediction_cutoff": cutoff.isoformat(), "visible_state": visible,
            "lineage": {"run_sha256": _digest(run), "cycle_step": cycle["step"],
                "public_before_sha256": _digest(public), "encoded_state_sha256": _digest(visible),
                "projection": "original_saved_public_snapshot_not_later_reprojection"}}
        if label is not None:
            row.update(outcome=label["outcome"], result_available_at=label["available_at"],
                       label_lineage=deepcopy(label))
        else:
            row["training_exclusion_reasons"] = label_reasons
        prefixes.append(row)
        if index == 0:
            continue
        previous = cycles[index-1]
        episode = by_step[previous["step"]]
        if (previous["public_state_after"] != public["state"]
                or previous["prediction_after"] != public["prediction"]["probabilities"]
                or parse_time(episode["available_at"]) > parse_time(before["decision_at"])):
            raise ValueError("After state is not the actual next visible snapshot")
        transitions.append({"schema": VERSION, "run_id": run["run_id"],
            "physical_match_id": identity["source_event_id"], "step": previous["step"],
            "before_prefix_id": prefixes[index-1]["prefix_id"], "after_prefix_id": prefix_id,
            "action": deepcopy(previous["selected_action"]),
            "response_status": episode["response"]["status"], "response_kind": episode["response"]["kind"],
            "decision_at": episode["decision_at"], "observed_at": episode["observed_at"],
            "available_at": episode["available_at"], "admission": deepcopy(previous["admission"]),
            "raw_sha256": episode["response"]["raw_sha256"],
            "selection_probability": episode["selection_probability"],
            "selection_scope": episode["selection_scope"], "selection_basis": episode["selection_basis"],
            "actual_cost": deepcopy(previous["actual_cost"]),
            "old_model_exclusion_reasons": deepcopy(episode["training_exclusion_reasons"]),
            "response_observation_eligible": episode["response_observation_eligible"],
            "outcome_attached_to_prefix_supervision_only": label is not None,
            "unobserved_alternative_returns_available": False})
    return prefixes, transitions


def _immediate_context_prefixes(run, episodes, identity, labels, as_of):
    """Project verified v4 before/actual-return pairs, never a later decision.

    Callers must first replay the original raw run. These additional checks
    retain its causal history while allowing the frozen predictor's age
    features to change between an immediate return and the next decision.
    A late return remains a billed transition with an explicit missing after;
    it cannot supply either a training prefix or a zero-valued gain target.
    """
    cycles = run["cycles"]
    if not cycles or any(c.get("selected_action") is None for c in cycles[:-1]):
        raise ValueError("Complete ordered context cycles are required")
    steps = [c["step"] for c in cycles]
    selected = [c["step"] for c in cycles if c.get("selected_action") is not None]
    if (steps != sorted(set(steps)) or sorted(e["step"] for e in episodes) != selected):
        raise ValueError("Complete actual response history is required, without dropped steps")
    by_step = {e["step"]: e for e in episodes}
    initial = cycles[0]["public_before"]
    mid = identity["original_match_id"]
    cutoff = parse_time(initial["state"]["prediction_cutoff"])
    kickoff = parse_time(initial["state"]["base"]["kickoff"])
    at = parse_time(as_of)
    if cutoff >= kickoff:
        raise ValueError("Context cutoff must precede kickoff")
    label, label_reasons = _label(mid, kickoff,
        [e["response"]["raw_sha256"] for e in episodes], labels, at)
    prefixes, transitions = [], []
    previous_after = None

    def append_prefix(public, cycle, name, actions):
        visible = encode_visible_state(public, actions)
        before = visible["before"]
        if (before["match_id"] != mid or parse_time(before["prediction_cutoff"]) != cutoff
                or parse_time(before["fixture"]["kickoff"]) != kickoff
                or not parse_time(before["decision_at"]) <= min(cutoff, at)
                or not before["history"]["order_matches_attempts"]):
            raise ValueError("Visible context prefix identity/history/clocks differ")
        index = len(prefixes)
        row = {"schema": VERSION, "prefix_id": f"{run['run_id']}:prefix:{index}",
            "match_id": mid, "physical_match_id": identity["source_event_id"],
            "run_id": run["run_id"], "prefix_index": index,
            "kickoff_at": kickoff.isoformat(), "prediction_cutoff": cutoff.isoformat(),
            "visible_state": visible,
            "lineage": {"run_sha256": _digest(run), "cycle_step": cycle["step"],
                "public_snapshot": name, "public_snapshot_sha256": _digest(public),
                "encoded_state_sha256": _digest(visible),
                "projection": "original_saved_public_snapshot_not_later_reprojection"}}
        if label is not None:
            row.update(outcome=label["outcome"], result_available_at=label["available_at"],
                       label_lineage=deepcopy(label))
        else:
            row["training_exclusion_reasons"] = deepcopy(label_reasons)
        prefixes.append(row)
        return row

    for index, cycle in enumerate(cycles):
        public = cycle["public_before"]
        decision = parse_time(public["decision_at"])
        if previous_after is not None:
            if (decision < parse_time(previous_after["decision_at"])
                    or any(public.get(k) != previous_after.get(k)
                           for k in ("state", "documents", "attempts", "budget"))):
                raise ValueError("Next decision changed visible contents/history without a return")
        action = cycle.get("selected_action")
        if action is None:
            # A terminal decision may occur after the deadline. Its raw record
            # is preserved, but it is not a prematch predictor training state.
            if decision <= cutoff:
                append_prefix(public, cycle, "public_before", cycle["generated_actions"])
            continue
        before = append_prefix(public, cycle, "public_before", cycle["generated_actions"])
        episode = by_step[cycle["step"]]
        returned = parse_time(cycle["returned_at"])
        observed, available = (parse_time(episode[k]) for k in ("observed_at", "available_at"))
        if (cycle["execution_status"] != "returned" or episode["action"] != action
                or parse_time(episode["decision_at"]) != decision
                or not decision <= observed <= available <= returned <= at):
            raise ValueError("Actual context response action/status/clocks differ")
        late = returned > cutoff
        overrun = bool(cycle.get("resource_overrun"))
        if overrun and not (any(cycle["charged_cost"].get(unit, 0.) > cap+1e-12
                                for unit, cap in public["budget"]["limits"].items())
                or any(cycle.get("exploration_charged_cost", {}).get(unit, 0.) > cap+1e-12
                       for unit, cap in cycle.get("exploration_limits", {}).items())):
            raise ValueError("A context budget overrun requires actually exceeded physical limits")
        allowed = _MODEL_ONLY_REASONS | _LABEL_REASONS
        if late:
            allowed |= {"response_not_operationally_available_before_cutoff",
                        "successful_response_not_admitted"}
        errors = set(episode["training_exclusion_reasons"])-allowed
        if errors or (not late and not episode["response_observation_eligible"]):
            raise ValueError("Invalid operational context prefix: "+", ".join(sorted(errors)))
        pair = {"schema": VERSION, "run_id": run["run_id"],
            "physical_match_id": identity["source_event_id"], "step": cycle["step"],
            "before_prefix_id": before["prefix_id"], "after_prefix_id": None,
            "action": deepcopy(action), "response_status": episode["response"]["status"],
            "response_kind": episode["response"]["kind"], "decision_at": episode["decision_at"],
            "observed_at": episode["observed_at"], "available_at": episode["available_at"],
            "returned_at": cycle["returned_at"], "admission": deepcopy(cycle["admission"]),
            "raw_sha256": episode["response"]["raw_sha256"],
            "selection_probability": episode["selection_probability"],
            "selection_scope": episode["selection_scope"], "selection_basis": episode["selection_basis"],
            "actual_cost": deepcopy(cycle["actual_cost"]),
            "old_model_exclusion_reasons": deepcopy(episode["training_exclusion_reasons"]),
            "response_observation_eligible": episode["response_observation_eligible"],
            "outcome_attached_to_prefix_supervision_only": label is not None,
            "unobserved_alternative_returns_available": False,
            "gain_timing_scope": "actual_immediate_return_not_next_decision"}
        if late or overrun:
            reason = "after_information_cutoff" if late else "actual_resource_budget_overrun"
            if (cycle.get("public_after") is not None or index != len(cycles)-1
                    or cycle.get("update", {}).get("status") != "not_updated"
                    or cycle.get("update", {}).get("reason") != reason):
                raise ValueError("A late/over-budget return cannot create an immediate predictor state")
            pair["gain_exclusion_reason"] = ("actual_return_after_prediction_cutoff" if late else reason)
            if overrun:
                pair["resource_overrun"] = True
                pair["charged_cost"] = deepcopy(cycle["charged_cost"])
                pair["physical_budget_limits"] = deepcopy(public["budget"]["limits"])
                pair["exploration_charged_cost"] = deepcopy(cycle.get("exploration_charged_cost", {}))
                pair["exploration_limits"] = deepcopy(cycle.get("exploration_limits", {}))
            transitions.append(pair)
            continue
        after = cycle.get("public_after")
        if (not isinstance(after, dict) or parse_time(after["decision_at"]) != returned
                or after["state"] != cycle["public_state_after"]
                or after["prediction"]["probabilities"] != cycle["prediction_after"]
                or after.get("budget", {}).get("spent") != cycle["charged_cost"]
                or after.get("budget", {}).get("limits") != public.get("budget", {}).get("limits")):
            raise ValueError("Immediate public after must bind the actual return, forecast and budget")
        expected_state = deepcopy(public["state"])
        expected_state["action_history"] = expected_state.get("action_history", [])+[{"action_id": action["action_id"]}]
        expected_attempt = {"action_id": action["action_id"], "information_key": action["information_key"],
            "status": episode["response"]["status"], "admission": cycle["admission"]["status"]}
        # Trajectory-enabled runs may retain additional causal audit fields
        # (attempt_id and the two observed clocks) on each attempt.  Preserve
        # those fields while checking the stable semantic projection required
        # by the older context-training schema.
        expected_attempts = public["attempts"]+[expected_attempt]
        actual_attempts = after["attempts"]
        attempts_match = (isinstance(actual_attempts, list)
            and len(actual_attempts) == len(expected_attempts)
            and all(all(actual.get(key) == expected.get(key) for key in expected)
                     for actual, expected in zip(actual_attempts, expected_attempts)))
        if (after["state"] != expected_state or not attempts_match):
            raise ValueError("Immediate return must append exactly the selected actual history")
        documents, original = after["documents"], public["documents"]
        if cycle["admission"]["status"] == "admitted":
            if (len(documents) != len(original)+1 or documents[:-1] != original
                    or any(documents[-1].get(k) != episode["response"].get(k)
                           for k in ("document_id", "raw_sha256"))):
                raise ValueError("Immediate after includes a different or unselected returned document")
        elif documents != original:
            raise ValueError("An unadmitted response cannot add or alter visible documents")
        immediate = append_prefix(after, cycle, "public_after", ())
        pair["after_prefix_id"] = immediate["prefix_id"]
        transitions.append(pair)
        previous_after = after
    return prefixes, transitions


def build_context_training_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None):
    """Validate original actor/result bytes and write an immutable prefix corpus.

    This does not fit a predictor. The new predictor's own readiness contract is
    reported; no old response-enumeration gate or topic-based filter is applied.
    """
    from football_research.acquisition.context_predictor import training_readiness
    started = datetime.now(timezone.utc)
    as_of = parse_time(started if as_of is None else as_of)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Context corpus output exists; choose a new immutable directory")
    if not isinstance(run_folders, (list, tuple)) or not run_folders or not result_archive:
        raise ValueError("Original run folders and independent raw result archive are required")
    runs, proofs, identities, inventory = _registered_runs(run_folders, previous_plans, as_of)
    bindings = _bindings([r["folder"] for r in runs]+[result_archive])
    joined = join_settlements([r["folder"] for r in runs], settlements=[], prior_bindings=proofs,
                              as_of=as_of, result_archive=result_archive)
    if joined["dataset"]["audit"]["rejected"]:
        raise ValueError("Raw gateway rejected an original run or receipt; do not shorten its history")
    by_run, labels = defaultdict(list), defaultdict(list)
    for row in joined["dataset"]["episodes"]:
        by_run[row["run_id"]].append(row)
    for check in joined["independent_result_verification"]:
        labels[check["match_id"]].extend(check.get("settlement_rows", []))
    prefixes, transitions, replays = [], [], []
    project_root = Path(__file__).resolve().parents[2]
    for item in runs:
        folder = Path(item["folder"])
        with TemporaryDirectory(prefix="fmctx-") as temporary:
            replay = replay_saved_run(folder, output=Path(temporary)/"r", project_root=project_root)
        replays.append({key: replay[key] for key in
                        ("run_id", "identical_replay", "raw_responses_reprojected", "replay_network_requests")})
        rows, pairs = _visible_prefixes(_read(folder/"run.json"), by_run[item["run_id"]],
                                        identities[item["match_id"]], labels, as_of)
        prefixes.extend(rows)
        transitions.extend(pairs)
    # Source inputs are immutable even when another task runs in this workspace.
    if bindings != _bindings([r["folder"] for r in runs]+[result_archive]):
        raise ValueError("Original input bytes changed during context conversion")
    records = [row for row in prefixes if "outcome" in row]
    pending = [row for row in prefixes if "outcome" not in row]
    readiness = training_readiness(records, as_of=as_of, evidence_origin="real_observational")
    by_event = defaultdict(list)
    for row in prefixes:
        by_event[row["physical_match_id"]].append(row)
    groups = [{"physical_match_id": event, "provider_match_ids": sorted({r["match_id"] for r in rows}),
        "run_ids": sorted({r["run_id"] for r in rows}), "prefix_ids": [r["prefix_id"] for r in rows],
        "prefixes": len(rows), "outcome": rows[0].get("outcome"),
        "kickoff_at": rows[0]["kickoff_at"], "split_unit": "complete_physical_match",
        "training_weight_contract": "one_total_weight_per_physical_match_after_feature_deduplication"}
        for event, rows in sorted(by_event.items())]
    report = {"schema": VERSION, "status": "converted", "readiness": readiness,
        "created_at": started.isoformat(), "as_of": as_of.isoformat(),
        "evidence_origin": "real_observational", "input_runs": len(runs),
        "physical_matches": len(groups), "prefixes": len(prefixes), "actual_transitions": len(transitions),
        "supervised_prefixes": len(records), "pending_prefixes": len(pending),
        "supervised_physical_matches": len({r["physical_match_id"] for r in records}),
        "response_statuses": dict(Counter(r["response_status"] for r in transitions)),
        "response_kinds": dict(Counter(r["response_kind"] for r in transitions)),
        "old_enumeration_or_unknown_role_gate_applied": False,
        "replay_acceptance": replays, "model_training_performed": False, "model": None,
        "realized_voi_benefit_proven": False, "network_requests": 0,
        "scope": "actual_logged_prefix_supervision_not_counterfactual_request_values",
        "historical_snapshots_modified": False,
        "split_requirement": "all_prefixes_aliases_runs_of_one_physical_match_share_one_time_partition",
        "records_sha256": _digest(records), "pending_sha256": _digest(pending),
        "transitions_sha256": _digest(transitions), "input_bindings": bindings,
        "registration_inputs_sha256": _digest(previous_plans)}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("prefix-records.json", records), ("pending-prefixes.json", pending),
                       ("actual-transitions.json", transitions), ("physical-groups.json", groups),
                       ("gateway-dataset.json", joined), ("registration-inputs.json", previous_plans),
                       ("identity-map.json", {"mappings": identities, "inventory": inventory}),
                       ("conversion-report.json", report)):
        _write(output/name, data)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--previous-plan", type=Path, action="append", required=True)
    parser.add_argument("--result-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_context_training_from_runs(args.run, [_read(p) for p in args.previous_plan],
                                              args.result_archive, args.output)
    print(json.dumps({key: report[key] for key in ("status", "readiness", "supervised_prefixes",
        "pending_prefixes", "model_training_performed", "network_requests")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
