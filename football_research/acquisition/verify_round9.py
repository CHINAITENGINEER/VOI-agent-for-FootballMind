"""Independent residual-calibration and policy audit for round nine.

The reference uses only frozen past calibration observations and public current
model EVSI. A descriptive cluster SE is not a single-match predictive interval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .state import parse_time
from .verify_round8 import close, require


BIN_BOUNDARIES = (.02, .05, .10)
UNITS = ("money", "requests", "latency_ms", "tokens")


def reference_bin(evsi):
    require(type(evsi) in (int, float) and math.isfinite(evsi) and evsi >= -1e-10, "Invalid public model EVSI")
    return int(np.searchsorted(BIN_BOUNDARIES, max(0., evsi), side="right"))


def acquired_from_state(state):
    """An observed missing return is purchased evidence, despite no numeric mask."""
    groups = state["groups"]
    acquired = []
    for aid, members in (("results_view", ("elo", "recent_form")), ("xg_view", ("xg",))):
        statuses = [groups[g]["status"] for g in members]
        if all(status in ("observed", "source_missing") for status in statuses):
            acquired.append(aid)
        else:
            require(all(status == "not_acquired" for status in statuses), "Invalid partial tool return")
    return tuple(acquired)


def reference_summary(records, model_evsi, *, min_support=12, shrinkage=20., multiplier=1.64):
    """Estimate a local residual correction with overlapping-parent cluster SE.

    Input rows must already be restricted to one (action, acquired-state) parent.
    Each parent has at most one transition from any given match.
    """
    require(len({r["match_id"] for r in records}) == len(records), "Duplicate match in parent cell")
    bucket = reference_bin(model_evsi)
    local = [r for r in records if reference_bin(r["model_evsi"]) == bucket]
    n, total = len(local), len(records)
    if total < min_support:
        return {"supported": False, "support": n, "parent_support": total,
                "residual_correction": None, "expected_gain": None, "standard_error": None,
                "lower_gain": None, "local_weight": 0.0, "time_blocks": 0}
    values = {r["match_id"]: float(r["delta"])-float(r["model_evsi"]) for r in records}
    require(all(math.isfinite(v) for v in values.values()), "Nonfinite residual")
    parent_mean = math.fsum(values.values())/total
    local_mean = math.fsum(values[r["match_id"]] for r in local)/n if n else parent_mean
    weight = n/(n+shrinkage) if n >= min_support else 0.
    correction = weight*local_mean + (1-weight)*parent_mean
    local_ids = {r["match_id"] for r in local}
    influence = defaultdict(float)
    for row in records:
        y, week, _ = parse_time(row["kickoff"]).isocalendar()
        residual = values[row["match_id"]]
        effect = (1-weight)*(residual-parent_mean)/total
        if row["match_id"] in local_ids and weight:
            effect += weight*(residual-local_mean)/n
        influence[y, week] += effect
    blocks = len(influence)
    se = math.sqrt(blocks/(blocks-1)*math.fsum(v*v for v in influence.values())) if blocks >= 2 else None
    expected = model_evsi+correction
    return {"supported": True, "support": n, "parent_support": total,
            "residual_correction": correction, "expected_gain": expected, "standard_error": se,
            "lower_gain": expected-multiplier*se if se is not None else None,
            "local_weight": weight, "time_blocks": blocks,
            "fallback": n < min_support, "bin": bucket}


def reference_decision(candidates, *, risk_level, money, requests, lambda_money):
    require(risk_level in ("mean", "conservative"), "Unsupported calibrated risk mode")
    metric = "expected_gain" if risk_level == "mean" else "lower_gain"
    usable = []
    for candidate in candidates:
        action = candidate["action"]
        gain = candidate.get(metric)
        if (not action.get("available", True) or not candidate["supported"] or gain is None
                or action["cost"]["money"] > money+1e-12
                or action["cost"]["requests"] > requests+1e-12):
            continue
        net = gain-lambda_money*action["cost"]["money"]
        usable.append((round(net, 12), action["action_id"], net))
    best = sorted(usable, key=lambda r:(-r[0], r[1]))[0] if usable else None
    aid = best[1] if best and best[0] > 0 else None
    return {"action_id": aid, "decision": "acquire" if aid else "stop",
            "selection_score": best[0] if aid else 0.0,
            "net_gain": best[2] if aid else 0.0}


def verify_value_cells(calibrator):
    """Rebuild every stored cell directly from its provenance-bearing rows."""
    parents = defaultdict(list)
    identities = set()
    for row in calibrator.records:
        key = row["action_id"], tuple(row["acquired_action_ids"])
        identity = row["match_id"], key
        require(identity not in identities, "Duplicate fitted calibration transition")
        identities.add(identity)
        require(json.loads(row["context_key"]) == [key[0], list(key[1])], "Calibration context differs")
        require(row["evsi_bin"] == reference_bin(row["model_evsi"]), "Calibration bucket differs")
        close(row["residual"], row["gain"]-row["model_evsi"], "Stored fitted residual")
        parents[key].append({**row, "delta": row["gain"]})
    expected_keys = {(key, bucket) for key in parents for bucket in range(4)}
    actual_keys, summaries = set(), {}
    representatives = (.01, .035, .075, .20)
    for encoded, actual in calibrator.cells.items():
        context, bucket = json.loads(encoded)
        action, acquired = json.loads(context)
        key = action, tuple(acquired)
        require((key, bucket) not in actual_keys and (key, bucket) in expected_keys, "Unknown or duplicate calibration cell")
        actual_keys.add((key, bucket))
        representative = representatives[bucket]
        wanted = reference_summary(parents[key], representative, min_support=calibrator.min_support,
                                   shrinkage=calibrator.shrinkage, multiplier=calibrator.risk_multiplier)
        for field in ("support", "parent_support", "time_blocks"):
            require(actual[field] == wanted[field], "Cell count differs: " + field)
        local_ids = {r["match_id"] for r in parents[key] if reference_bin(r["model_evsi"]) == bucket}
        parent_ids = {r["match_id"] for r in parents[key]}
        require(set(actual["local_match_ids"]) == local_ids and set(actual["parent_match_ids"]) == parent_ids, "Cell membership differs")
        for actual_field, wanted_field in (("residual_mean", "residual_correction"), ("standard_error", "standard_error"), ("local_weight", "local_weight")):
            if wanted[wanted_field] is None:
                require(actual[actual_field] is None, "Invented unsupported numerical estimate")
            else:
                close(actual[actual_field], wanted[wanted_field], "Independent cell " + actual_field)
        summaries[key, bucket] = wanted
    require(actual_keys == expected_keys, "Missing calibration cells")
    return parents, summaries


def check_estimate(actual, wanted, evsi):
    """A support count or a lower bound cannot be changed silently in a trace."""
    close(actual["model_evsi"], evsi, "Independent model EVSI")
    for field in ("expected_gain", "lower_gain", "standard_error"):
        if wanted[field] is None:
            require(actual[field] is None, "Unsupported gain/uncertainty became numeric")
        else:
            close(actual[field], wanted[field], "Independent calibrated " + field)
    for field in ("support", "parent_support", "time_blocks"):
        require(actual[field] == wanted[field], "Recorded support differs: " + field)
    close(actual["local_weight"], wanted["local_weight"], "Recorded shrinkage weight")
    require(actual["evsi_bin"] == reference_bin(evsi), "Recorded EVSI bucket differs")


def chronological_partition(samples, fraction=.2):
    """Independently reproduce the prespecified time-group split and purge."""
    matches = {s["match_id"]: s for s in samples}
    ordered = sorted(matches.values(), key=lambda r:(parse_time(r["kickoff"]),r["match_id"]))
    require(len(ordered) >= 3, "Too few calibration matches")
    boundary = parse_time(ordered[max(1, int(len(ordered)*(1-fraction)))]["kickoff"])
    tail = [s for s in samples if parse_time(s["kickoff"]) >= boundary]
    prediction_boundary = min(parse_time(s["state"]["prediction_cutoff"]) for s in tail)
    train = [s for s in samples if parse_time(s["kickoff"]) < boundary
             and parse_time(s["result_available_at"]) < prediction_boundary]
    return train, tail, prediction_boundary


def transition_identity(sample):
    return sample["match_id"], sample["action"]["action_id"], tuple(sorted(sample["acquired_action_ids"]))


def fitted_identity(record):
    return record["match_id"], record["action_id"], tuple(sorted(record["acquired_action_ids"]))


def verify_fitted_provenance(calibrator, train, tail, prediction_boundary, cutoff):
    expected = {transition_identity(s): s for s in train}
    actual = {fitted_identity(r): r for r in calibrator.records}
    require(set(expected) == set(actual) and len(actual) == len(calibrator.records), "Calibration fitting membership differs")
    manifest = calibrator.manifest
    require(set(manifest["training_match_ids"]) == {r["match_id"] for r in train}, "Calibration train manifest differs")
    require(set(manifest["calibration_match_ids"]) == {r["match_id"] for r in tail}, "Diagnostic tail manifest differs")
    require(parse_time(manifest["trained_as_of"]) < prediction_boundary, "Fitted calibration time overlaps diagnostic tail")
    require(parse_time(manifest["diagnostics_as_of"]) == cutoff, "Diagnostic availability cutoff differs")
    for key, sample in expected.items():
        record = actual[key]
        for field in ("kickoff", "result_available_at", "oof_train_cutoff", "oof_manifest_ref", "oof_predictor_version"):
            require(record[field] == sample[field], "Fitted sample provenance differs: " + field)
        require(set(record["oof_train_ids"]) == set(sample["oof_train_ids"]), "Fitted OOF base training IDs differ")
        require(parse_time(record["result_available_at"]) <= parse_time(manifest["trained_as_of"]), "Fitted calibration label lookahead")
        require(record["prediction_cutoff"] == sample["state"]["prediction_cutoff"], "Fitted sample cutoff differs")
        close(record["gain"], sample["delta"], "Fitted new joint gain")
        close(record["model_evsi"], max(0., sample["model_evsi"]), "Fitted new joint EVSI")
        close(record["residual"], sample["delta"]-max(0.,sample["model_evsi"]), "Fitted new residual")
    parents, summaries = verify_value_cells(calibrator)
    diagnostics = calibrator.calibration["records"]
    indexed = {fitted_identity(r): r for r in diagnostics}
    require(set(indexed) == {transition_identity(r) for r in tail} and len(indexed) == len(diagnostics), "Diagnostic tail transition coverage differs")
    for sample in tail:
        record = indexed[transition_identity(sample)]
        key = sample["action"]["action_id"], tuple(sorted(sample["acquired_action_ids"]))
        wanted = reference_summary(parents.get(key, []), sample["model_evsi"], min_support=calibrator.min_support,
                                   shrinkage=calibrator.shrinkage, multiplier=calibrator.risk_multiplier)
        check_estimate(record, wanted, sample["model_evsi"])
        close(record["gain"], sample["delta"], "Diagnostic empirical gain")
    return parents, summaries


def independent_state(model, source, actions, acquired):
    from .experiment import state_after
    from .verify_round8 import rebuild_joint, probability_checks, marginal

    state = state_after(source, actions, acquired)
    cube, observed = rebuild_joint(model, state)
    require(set(observed) == set(acquired), "Independent acquired return state differs")
    available = [a["action_id"] for a in actions if a["action_id"] not in observed]
    values = probability_checks(cube, available)
    prediction = model.predict_partial(state)
    for a,b in zip(prediction["probabilities"], marginal(cube)):
        close(a,b,"Independent joint probability")
    return state, prediction, values


class IndependentEvaluator:
    """Cache only deterministic public states; never call the new calibrator."""
    def __init__(self, sources, actions, models, calibrators, parents):
        self.sources, self.actions, self.models = sources, actions, models
        self.calibrators, self.parents = calibrators, parents
        self.states, self.estimates = {}, {}

    def state(self, fold, mid, acquired):
        key = fold, mid, tuple(sorted(acquired))
        if key not in self.states:
            self.states[key] = independent_state(self.models[fold], self.sources[mid], self.actions, acquired)
        return self.states[key]

    def candidates(self, fold, mid, acquired):
        key = fold, mid, tuple(sorted(acquired))
        if key not in self.estimates:
            _, prediction, values = self.state(fold, mid, acquired)
            calibrator, output = self.calibrators[fold], []
            for action in self.actions:
                aid = action["action_id"]
                if aid in acquired:
                    continue
                model_evsi = max(0., values[aid]["evsi"])
                summary = reference_summary(self.parents[fold].get((aid, tuple(sorted(acquired))), []), model_evsi,
                    min_support=calibrator.min_support, shrinkage=calibrator.shrinkage, multiplier=calibrator.risk_multiplier)
                output.append({**summary, "model_evsi": model_evsi, "action": action})
            self.estimates[key] = output
        return self.estimates[key]

    def episode(self, fold, mid, *, risk_level, budget_money, lambda_money, request_budget):
        from .experiment import losses, outcome

        acquired, money, requests = [], 0., 0.
        for _ in range(len(self.actions)+1):
            candidates = self.candidates(fold, mid, acquired)
            decision = reference_decision(candidates,risk_level=risk_level,money=budget_money-money,
                                          requests=request_budget-requests,lambda_money=lambda_money)
            aid = decision["action_id"]
            if aid is None:
                break
            action = next(a for a in self.actions if a["action_id"] == aid)
            acquired.append(aid)
            money += action["cost"]["money"]
            requests += action["cost"]["requests"]
        _, prediction, _ = self.state(fold, mid, acquired)
        return {"actions": acquired, "money": money, "requests": requests,
                "probabilities": prediction["probabilities"],
                **losses(prediction["probabilities"],outcome(self.sources[mid]))}


def verify_calibrated_trace(row, trace, source, actions, evaluator, request_budget):
    from .data import ControlledEpisode
    from .experiment import losses, outcome

    fold, mid = row["fold"], row["match_id"]
    risk_level = "conservative" if row["strategy"] == "calibrated-lcb" else "mean"
    require(row["strategy"] in ("calibrated-mean","calibrated-lcb"), "Unknown calibrated strategy")
    env, spent = ControlledEpisode(source, actions), dict.fromkeys(UNITS,0.)
    count, candidate_count = 0, 0
    events = trace["events"]
    require(bool(events),"Empty calibrated trace")
    for step,event in enumerate(events):
        require(event["step"] == step, "Noncontiguous calibrated trace")
        state = {**env.visible_state(),"action_history":[{"action_id":aid,"status":"acquired_or_source_missing"} for aid in env.performed_actions]}
        require(state == event["visible_state"], "Unpurchased content or invalid acquired state")
        _, prediction, _ = evaluator.state(fold,mid,env.performed_actions)
        for a,b in zip(event["prediction"]["probabilities"],prediction["probabilities"]):
            close(a,b,"Calibrated policy retains frozen predictor")
        decision = event["decision"]
        candidates = evaluator.candidates(fold,mid,env.performed_actions)
        reference = reference_decision(candidates,risk_level=risk_level,money=row["budget_money"]-spent["money"],
            requests=request_budget-spent["requests"],lambda_money=row["lambda_money"])
        require(decision["action_id"] == reference["action_id"] and decision["decision"] == reference["decision"], "Independent calibrated action differs")
        if env.legal_actions():
            require(decision["horizon"] == 1 and decision["risk_level"] == risk_level, "Wrong calibrated horizon/risk")
            for unit in UNITS:
                close(decision["physical_budget"]["spent"][unit],spent[unit],"Calibrated recorded spent")
            close(decision["physical_budget"]["limits"]["money"],row["budget_money"],"Calibrated money ceiling")
            close(decision["physical_budget"]["limits"]["requests"],request_budget,"Calibrated request ceiling")
            indexed={c["action_id"]:c for c in decision["candidates"]}
            require(len(indexed)==len(decision["candidates"])==len(candidates),"Calibrated candidate coverage differs")
            for candidate in candidates:
                action=candidate["action"]
                actual=indexed[action["action_id"]]
                affordable=(action["cost"]["money"] <= row["budget_money"]-spent["money"]+1e-12 and
                            action["cost"]["requests"] <= request_budget-spent["requests"]+1e-12)
                if not affordable:
                    require(not actual["feasible"] and actual["reason"]=="physical_budget_exhausted", "Wrong unaffordable gate")
                    continue
                check_estimate(actual,candidate,candidate["model_evsi"])
                penalty=row["lambda_money"]*action["cost"]["money"]
                close(actual["cost_penalty"],penalty,"Calibrated cost conversion")
                gain=candidate["expected_gain" if risk_level=="mean" else "lower_gain"]
                require(actual["feasible"] == (gain is not None),"Unsupported estimate became feasible")
                if gain is not None:
                    close(actual["net_gain"],gain-penalty,"Calibrated net gain")
                    close(actual["selection_score"],round(gain-penalty,12),"Calibrated deterministic score")
                candidate_count += 1
        if decision["decision"] == "stop":
            require(step==len(events)-1 and "charged_cost" not in event,"Charged or continued after calibrated stop")
            require(row["stop_reason"]==decision["reason"],"Calibrated stop reason differs")
            break
        aid=decision["action_id"]
        action=next(a for a in env.legal_actions() if a["action_id"]==aid)
        require(event["charged_cost"]==action["cost"],"Calibrated charge differs")
        for unit in UNITS:
            spent[unit]+=action["cost"].get(unit,0.)
        require(spent["money"]<=row["budget_money"]+1e-10 and spent["requests"]<=request_budget,"Calibrated physical budget exceeded")
        env.reveal(aid)
        count += 1
    require(events[-1]["decision"]["decision"]=="stop", "Calibrated episode missing STOP")
    require(row["actions"]==list(env.performed_actions) and row["final_mask"]==prediction["mask"],"Calibrated final evidence differs")
    require(row["final_probabilities"]==prediction["probabilities"],"Calibrated final probabilities differ")
    for unit in UNITS:
        close(row["cost"][unit],spent[unit],"Calibrated total charge")
    for name,value in losses(prediction["probabilities"],outcome(source)).items():
        close(row[name],value,"Calibrated source score "+name)
    require(row["actual_external_requests"]==row["actual_llm_tokens"]==0,"Unexpected calibrated external use")
    return {"events":len(events),"charges":count,"candidates":candidate_count}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify(run, bundle, earlier_bundle):
    """Audit new models, fitting rows, decisions, and frozen-reference parity."""
    from datetime import timedelta
    from football_research.evaluation.splits import rolling_origin_splits
    from .contextual_partial import ContextualPartialPredictor
    from .data import load_episodes
    from .experiment import losses, outcome, state_after
    from .joint_returns import JointReturnPredictor, extract_oof_contexts
    from .round9_residual import ResidualEVSICalibrator
    from .verify_round4 import _rows
    from .verify_round6 import KEYS, verify_aggregate_metrics
    from .verify_round7 import verify_episode_coverage, verify_oof_sample
    from .verify_round8 import verify_training

    run = Path(run)
    read = lambda name: json.loads((run/name).read_text(encoding="utf-8"))
    report, design = read("report.json"), read("design.json")
    prior7, prior8 = Path(design["source_round7"]), Path(design["source_round8"])
    require(report["status"] == "OK", "Incomplete research run")
    require(report["implementation_sha256"] == read("implementation.json"), "Source manifest differs")
    for name, expected in read("implementation.json").items():
        require(digest(Path(__file__).parent/name) == expected, "Frozen implementation changed: "+name)
    for artifact in report["model_artifacts"]:
        require(digest(run/artifact["path"]) == artifact["sha256"], "Artifact hash differs")
    require(read("source-artifacts.json") == report["source_artifacts_sha256"], "Upstream manifest differs")
    for source, path in (("round7", prior7), ("round8", prior8)):
        for name, expected in read("source-artifacts.json")[source].items():
            require(digest(path/name) == expected, "Frozen upstream artifact changed")
    prior_report = json.loads((prior8/"report.json").read_bytes())
    prior_folds = {f["fold"]: f for f in prior_report["folds"]}
    prior_design = json.loads((prior8/"design.json").read_bytes())
    for field in ("joint_model", "validation_start", "money_budgets", "request_budget", "random_seeds",
                  "lambda_money_grid", "validation_utility_money_weight", "lead_hours"):
        require(design[field] == prior_design[field], "Prespecified reference design changed: "+field)
    require(design["residual_calibration"]["bin_edges"] == list(BIN_BOUNDARIES), "Fixed bin edges changed")
    prior_audit_path = prior8.parent/(prior8.name+"-verification.json")
    prior_audit = json.loads(prior_audit_path.read_bytes())
    require(prior_audit["status"] == "VERIFIED", "Frozen reference lacks completed independent verification")
    for name, expected in prior_audit["artifacts"].items():
        require(digest(prior8/name) == expected, "Previously verified reference changed")
    require(digest(Path(__file__).with_name("verify_round8.py")) == prior_audit["verifier_sha256"], "Prior audit implementation changed")
    require(digest(run/"catalog.json") == digest(prior8/"catalog.json"), "Action catalog changed")
    actions = read("catalog.json")["actions"]
    current, ca = load_episodes(bundle, lead_hours=design["lead_hours"])
    earlier, ea = load_episodes(earlier_bundle, lead_hours=design["lead_hours"])
    for name, audit in (("current", ca), ("earlier", ea)):
        require(read("data-audit.json")[name]["platform_manifest_sha256"] == audit["platform_manifest_sha256"], "Input export changed")
    all_rows = earlier+current
    sources = {r["match_id"]: r for r in all_rows}
    require(len(sources) == len(all_rows) == report["input_matches"], "Input identity coverage differs")

    contexts = {r["match_id"]: r for r in _rows(run/"training-oof-contexts.jsonl")}
    require(contexts == extract_oof_contexts(_rows(prior7/"oof-supervision.jsonl")), "Base OOF prior extraction differs")
    old_oof = json.loads((prior7/"oof-manifest.json").read_bytes())["folds"]
    checked_contexts = set()
    for entry in old_oof:
        path = prior7/entry["artifact"]
        require(digest(path) == entry["sha256"], "Original OOF base hash differs")
        base = ContextualPartialPredictor.load(path)
        ids = set(entry["training_match_ids"])
        require(ids == set(base.metadata["training_match_ids"]), "Original OOF train membership differs")
        for mid in ids:
            require(parse_time(sources[mid]["result_available_at"]) <= parse_time(entry["cutoff"]), "Base OOF label lookahead")
        for mid in entry["prediction_match_ids"]:
            require(mid not in checked_contexts and mid not in ids, "Base OOF target repeated or trained")
            checked_contexts.add(mid)
            ctx = contexts[mid]
            require(set(ctx["oof_train_ids"]) == ids and ctx["oof_train_cutoff"] == entry["cutoff"], "Base OOF provenance differs")
            require(parse_time(entry["cutoff"]) < parse_time(sources[mid]["prediction_cutoff"]), "Base OOF cutoff lookahead")
            require(ctx["oof_predictor_version"] == base.model_version == entry["model_version"], "Base OOF version differs")
            p = base.predict_partial(state_after(sources[mid], actions, ())) ["probabilities"]
            for a,b in zip(p,ctx["probabilities"]):
                close(a,b,"Original OOF empty prior")
    require(checked_contexts == set(contexts), "Base OOF coverage differs")

    new_oof = read("oof-manifest.json")
    oof_models, oof_targets, chunk_ids = {}, {}, set()
    oof_training_count = 0
    for entry in new_oof["folds"]:
        index = entry["source_chunk"]
        require(index not in chunk_ids, "Duplicate OOF source chunk")
        chunk_ids.add(index)
        original = old_oof[index]
        for field in ("cutoff", "prediction_match_ids", "training_match_ids"):
            require(entry[field] == original[field], "Forward OOF partition changed")
        require(entry["base_training_match_ids"] == original["training_match_ids"], "OOF base membership changed")
        joint_path, base_path = run/entry["artifact"], run/entry["base_artifact"]
        require(digest(joint_path) == entry["sha256"], "OOF joint hash differs")
        require(digest(base_path) == entry["base_sha256"] == original["sha256"], "OOF base bytes changed")
        model = JointReturnPredictor.load(joint_path)
        require(model.base_predictor.to_state() == ContextualPartialPredictor.load(base_path).to_state(), "OOF embedded base changed")
        require(model.model_version == entry["model_version"], "OOF joint version differs")
        require(model.bandwidth == .2 and model.dirichlet_mass == 5., "New OOF joint hyperparameters changed")
        oof_training_count += verify_training(model, entry, sources, contexts, parse_time(entry["cutoff"]))
        oof_models[index] = model
        for mid in entry["prediction_match_ids"]:
            require(mid not in oof_targets and mid not in entry["base_training_match_ids"], "Joint OOF target repeats or trains itself")
            oof_targets[mid] = entry
    for skipped in new_oof["skipped_chunks"]:
        index = skipped["source_chunk"]
        require(index not in chunk_ids, "Skipped OOF chunk also fitted")
        chunk_ids.add(index)
        original = old_oof[index]
        require(skipped["prediction_match_ids"] == original["prediction_match_ids"], "Skipped targets changed")
        require(not(set(original["training_match_ids"]) & set(contexts)), "Supported OOF chunk incorrectly skipped")
    require(chunk_ids == set(range(len(old_oof))), "Missing OOF source chunks")
    samples, identities, oof_state_cache = list(_rows(run/"oof-supervision.jsonl")), set(), {}
    for sample in samples:
        mid = sample["match_id"]
        entry = oof_targets[mid]
        model = oof_models[entry["source_chunk"]]
        identity = verify_oof_sample(sample,sources[mid],model,actions,entry["base_training_match_ids"],entry["cutoff"])
        require(identity not in identities, "Duplicate forward joint transition")
        identities.add(identity)
        require(sample["oof_manifest_ref"] == "sha256:"+entry["sha256"], "New gain target artifact provenance differs")
        require(set(sample["joint_training_match_ids"]) == set(entry["joint_training_match_ids"]), "New gain target q-membership differs")
        acquired = tuple(sorted(sample["acquired_action_ids"]))
        key = entry["source_chunk"], mid, acquired
        if key not in oof_state_cache:
            oof_state_cache[key] = independent_state(model,sources[mid],actions,acquired)
        evsi = max(0., oof_state_cache[key][2][sample["action"]["action_id"]]["evsi"])
        close(sample["model_evsi"],evsi,"New joint OOF independent EVSI")
        close(sample["residual"],sample["delta"]-evsi,"New joint OOF residual")
    required_identities = {(mid,acquired,aid) for mid in oof_targets
        for acquired,aid in (((),"results_view"),((),"xg_view"),(("results_view",),"xg_view"),(("xg_view",),"results_view"))}
    require(identities == required_identities, "New forward transition coverage differs")
    require(len(samples) == new_oof["samples"] == report["oof_samples"], "OOF sample count differs")
    require(len(oof_targets) == new_oof["independent_matches"] == report["oof_matches"], "OOF match count differs")
    print(f"round9 verified new OOF: {len(samples)} transitions / {len(oof_models)} models",flush=True)

    splits = list(rolling_origin_splits(all_rows,initial_validation_start=design["validation_start"],
        validation_window=timedelta(days=30),test_window=timedelta(days=30)))
    models, calibrators, parents, fold_ids = {}, {}, {}, {}
    cells_count = fitted_count = tail_count = 0
    for fold in report["folds"]:
        number = fold["fold"]
        split, old = splits[number], prior_folds[number]
        require(fold["split"] == split.manifest(), "Outer temporal split changed")
        for field, records in (("training_match_ids",split.train),("validation_match_ids",split.validation),("test_match_ids",split.test)):
            require(set(fold[field]) == set(old[field]) == {r["match_id"] for r in records}, "Outer identities changed")
        fold_ids[number] = set(fold["test_match_ids"])
        path = run/fold["joint_model"]["path"]
        require(digest(path) == digest(prior8/old["joint_model"]["path"]) == fold["joint_model"]["sha256"], "Outer predictor changed")
        model = JointReturnPredictor.load(path)
        require(model.model_version == fold["joint_model"]["version"], "Outer joint version changed")
        verify_training(model,fold,sources,contexts,split.train_cutoff)
        calpath = run/fold["calibrator"]["path"]
        require(digest(calpath) == fold["calibrator"]["sha256"], "Outer calibrator hash differs")
        calibrator = ResidualEVSICalibrator.load(calpath,model=model,model_sha256=digest(path))
        require(calibrator.version == fold["calibrator"]["version"] and calibrator.manifest == fold["calibrator"]["manifest"], "Outer calibrator identity differs")
        require((calibrator.min_support,calibrator.shrinkage,calibrator.risk_multiplier) == (12,20.,1.64), "Residual hyperparameters changed")
        eligible = [s for s in samples if s["match_id"] in set(fold["training_match_ids"])
                    and parse_time(s["result_available_at"]) <= split.train_cutoff]
        train, tail, boundary = chronological_partition(eligible)
        p,_ = verify_fitted_provenance(calibrator,train,tail,boundary,split.train_cutoff)
        metadata = fold["residual_split"]
        require(metadata == calibrator.manifest["metadata"], "Split metadata differs")
        require(set(metadata["purged_match_ids"]) == {s["match_id"] for s in eligible}-{s["match_id"] for s in train+tail}, "Purge membership differs")
        require(parse_time(metadata["fit_cutoff"]) == boundary-timedelta(microseconds=1), "Calibration fit cutoff differs")
        models[number],calibrators[number],parents[number] = model,calibrator,p
        cells_count += len(calibrator.cells)
        fitted_count += len(train)
        tail_count += len(tail)
    require(set(models) == set(prior_folds), "Outer fold coverage differs")
    evaluator = IndependentEvaluator(sources,actions,models,calibrators,parents)
    validation_count = 0
    for fold in report["folds"]:
        number = fold["fold"]
        for key,selection in fold["lambda_selection"].items():
            strategy,budget = key.split("/")
            if not strategy.startswith("calibrated-"):
                require(selection == prior_folds[number]["lambda_selection"][key], "Raw lambda selection changed")
                continue
            require(set(selection["validation_match_ids"]) == set(fold["validation_match_ids"]), "Lambda uses wrong validation partition")
            require([r["lambda_money"] for r in selection["candidates"]] == design["lambda_money_grid"], "Lambda grid changed")
            scores = []
            for candidate in selection["candidates"]:
                lam = candidate["lambda_money"]
                episodes = [evaluator.episode(number,mid,risk_level="mean" if strategy.endswith("mean") else "conservative",
                    budget_money=float(budget),lambda_money=lam,request_budget=design["request_budget"]) for mid in fold["validation_match_ids"]]
                utility = float(np.mean([r["log_loss"]+design["validation_utility_money_weight"]*r["money"] for r in episodes]))
                close(candidate["validation_utility"],utility,"Independent calibrated lambda utility")
                require(candidate["validation_matches"] == len(episodes), "Lambda sample count differs")
                validation_count += len(episodes)
                scores.append((utility,-lam,lam))
            close(selection["selected"],min(scores)[2],"Independent calibrated lambda selection")
        print(f"round9 verified calibration cells and validation policy: fold {number}",flush=True)

    key_for = lambda row: tuple(row[k] for k in KEYS)
    rows = list(_rows(run/"episodes.jsonl"))
    verify_episode_coverage(rows,fold_ids,design["strategies"],design["money_budgets"],design["random_seeds"])
    require(len(rows) == report["episode_runs"], "Episode count differs")
    traces = {key_for(t):t for t in _rows(run/"trajectories.jsonl")}
    require(len(traces) == len(rows), "Trace identity count differs")
    old_rows = {key_for(r):r for r in _rows(prior8/"episodes.jsonl")}
    raw_keys = set(old_rows)
    totals = dict.fromkeys(("events","charges","candidates"),0)
    calibrated_count = 0
    for row in rows:
        identity = key_for(row)
        require(identity in traces, "Missing episode trace")
        number,mid = row["fold"],row["match_id"]
        require(row["model_version"] == models[number].model_version, "Episode joint version differs")
        if not row["strategy"].startswith("calibrated-"):
            old = old_rows[identity]
            ignore = {"predictor_latency_ms","estimator_latency_ms"}
            require({k:v for k,v in row.items() if k not in ignore} == {k:v for k,v in old.items() if k not in ignore}, "Raw episode changed from verified round8")
            continue
        require(row["gain_version"] == calibrators[number].version, "Episode calibration version differs")
        fold = next(f for f in report["folds"] if f["fold"] == number)
        close(row["lambda_money"],fold["lambda_selection"][f"{row['strategy']}/{row['budget_money']:g}"]["selected"],"Episode lambda differs")
        count = verify_calibrated_trace(row,traces[identity],sources[mid],actions,evaluator,design["request_budget"])
        for field in totals:
            totals[field] += count[field]
        calibrated_count += 1
    seen_raw = set()
    for previous in _rows(prior8/"trajectories.jsonl"):
        identity = key_for(previous)
        require(identity not in seen_raw and previous == traces[identity], "Raw decision trace differs from independently verified round8")
        seen_raw.add(identity)
    require(seen_raw == raw_keys, "Raw trace coverage differs")
    verify_aggregate_metrics(rows,report["metrics"])
    for metric in report["metrics"].values():
        close(metric["utility_loss"],metric["log_loss"]+design["validation_utility_money_weight"]*metric["money"],"Aggregate utility")

    diagnostics = list(_rows(run/"update-gain-diagnostics.jsonl"))
    seen_diagnostics = set()
    for record in diagnostics:
        fold,mid,aid = record["fold"],record["match_id"],record["action_id"]
        acquired = tuple(sorted(record["acquired_action_ids"]))
        identity = fold,mid,acquired,aid
        require(identity not in seen_diagnostics and mid in fold_ids[fold], "Duplicate or off-test diagnostic")
        seen_diagnostics.add(identity)
        candidate = next(c for c in evaluator.candidates(fold,mid,acquired) if c["action"]["action_id"] == aid)
        check_estimate(record,candidate,candidate["model_evsi"])
        before = evaluator.state(fold,mid,acquired)[1]["probabilities"]
        after = evaluator.state(fold,mid,(*acquired,aid))[1]["probabilities"]
        for field, probabilities in (("prediction_before",before),("prediction_after",after)):
            for a,b in zip(record[field],probabilities):
                close(a,b,"Diagnostic independent prediction")
        gain = losses(before,outcome(sources[mid]))["log_loss"]-losses(after,outcome(sources[mid]))["log_loss"]
        close(record["gain"],gain,"Diagnostic actual update gain")
    expected_diagnostics = {(fold,mid,acquired,aid) for fold,ids in fold_ids.items() for mid in ids
        for acquired,aid in (((),"results_view"),((),"xg_view"),(("results_view",),"xg_view"),(("xg_view",),"results_view"))}
    require(seen_diagnostics == expected_diagnostics, "Complete test transition coverage differs")
    summary = report["update_gain_diagnostics"]
    for field,values in (
        ("mean_model_evsi",[r["model_evsi"] for r in diagnostics]),
        ("mean_estimated_gain",[r["expected_gain"] for r in diagnostics]),
        ("mean_realized_gain",[r["gain"] for r in diagnostics]),
        ("mean_squared_error",[(r["gain"]-r["expected_gain"])**2 for r in diagnostics]),
        ("raw_model_mean_squared_error",[(r["gain"]-r["model_evsi"])**2 for r in diagnostics])):
        close(summary[field],float(np.mean(values)),"Aggregate gain diagnostic")
    artifacts = {name:digest(run/name) for name in ("report.json","design.json","catalog.json","episodes.jsonl",
        "trajectories.jsonl","oof-manifest.json","oof-supervision.jsonl","training-oof-contexts.jsonl","update-gain-diagnostics.jsonl")}
    return {"status":"VERIFIED","round":9,"test_matches":len(set().union(*fold_ids.values())),
        "episode_runs":len(rows),"frozen_round8_episode_trace_parity":len(raw_keys),
        "calibrated_episodes_independently_replayed":calibrated_count,"calibrated_visible_events":totals["events"],
        "calibrated_charges":totals["charges"],"calibrated_affordable_candidates":totals["candidates"],
        "base_oof_priors_recomputed":len(checked_contexts),"new_joint_oof_models":len(oof_models),
        "new_joint_oof_training_records":oof_training_count,"new_joint_oof_transitions":len(samples),
        "new_joint_oof_matches":len(oof_targets),"calibration_cells_recomputed":cells_count,
        "calibration_fitted_transitions_across_folds":fitted_count,"diagnostic_tail_transitions_across_folds":tail_count,
        "validation_episodes_independently_recomputed":validation_count,"complete_test_transitions":len(diagnostics),
        "independent_joint_public_states":len(evaluator.states)+len(oof_state_cache),
        "artifacts":artifacts,"model_artifacts":len(report["model_artifacts"]),
        "frozen_reference_audit_sha256":digest(prior_audit_path),"verifier_sha256":digest(__file__),
        "external_requests":0,"llm_tokens":0,
        "scope":"Implementation/provenance verification on reused historical games; not calibration coverage or scientific success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","bundle","earlier-bundle","output"):
        parser.add_argument("--"+name,required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = verify(args.run,args.bundle,args.earlier_bundle)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="artifacts"},ensure_ascii=True))
