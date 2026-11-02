"""Replay all 40 synthetic return patterns without revealing unpurchased evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from uuid import uuid4

from .finite_information_bridge import VERSION,MODE,CUTOFF,CLASSES,FiniteObservationEnvironment,FiniteInformationWorkflow,make_finite_backend,build_finite_cards
from .verify_round8_joint import _hash,_equal,_require


DESIGN_SHA256="aa70d9bd6ac92c04722935cdf021d29deaaa811e6f496f5d40fd420aab37a64b"
COUNTS={"gated_stop":6,"gated_route":18,"complementary":9,"redundant":3,"irrelevant":4}


def select_patterns(scenario):
    """Group by complete tuple only; keep every world for offline parity checks."""
    grouped={}
    for world in scenario["worlds"]:
        if world["mass"]<=0:continue
        key=tuple(sorted(world["observations"].items()))
        grouped.setdefault(key,[]).append(world["world_id"])
    return [(dict(key),world_ids) for key,world_ids in sorted(grouped.items())]


def observed_state(visible):
    return {"observations":{aid:g["values"]["category"] for aid,g in visible["groups"].items()},
            "action_history":deepcopy(visible["action_history"])}


def compare_run(run,events):
    """All model branches are compared, but only actual receipts advance history."""
    by_step={e["step"]:e for e in events}
    count=branches=predictions=receipts=0
    observed={}
    for event in run["events"]:
        kind=event["type"]
        if kind=="acquisition_result":
            actual=event["data"]["result"]
            expected=by_step[len(observed)]["receipt"]
            _require(actual["action_id"]==expected["action_id"] and actual["observation"]==expected["observation"],"Actual categorical return differs")
            _equal(actual["cost_vector"],expected["cost"],"Actual single-query cost vector")
            _equal(actual["cost"]["amount"],expected["cost"]["money"],"Actual single-query charge")
            observed[actual["action_id"]]=actual["observation"]
            receipts+=1
        visible=event["snapshot"].get("visible_state")
        if visible is not None:
            _equal(observed_state(visible)["observations"],observed,"Only already received evidence appears in snapshots")
        if kind=="prediction_updated":
            ref=by_step[len(observed)]
            prediction=event["snapshot"]["prediction"]
            _equal([prediction["probabilities"][k] for k in CLASSES],ref["prediction"]["probabilities"],"Actual posterior")
            _equal(observed_state(visible),ref["visible_state"],"Actual model public state")
            predictions+=1
        elif kind=="value_estimated":
            plan=by_step[len(observed)]["decision"]
            expected={c["action_id"]:c for c in plan["candidates"] if c["feasible"]}
            actual={c["action_id"]:c["value"] for c in event["data"]["candidates"] if "value" in c}
            _require(set(actual)==set(expected),"Compared different legal finite candidates")
            for aid,value in actual.items():
                native=value["research_estimate"];ref=expected[aid]
                _equal({k:native[k] for k in ref},ref,"Native candidate and complete model branches")
                _equal(value["expected_loss_improvement"],ref["expected_loss_improvement"],"Planned total gain")
                _equal(value["cost_in_loss_units"],ref["expected_plan_penalty"],"Planned total penalty")
                _equal(value["net_gain"],ref["selection_score"],"Rounded choice score")
                _require(native["planned_action"]==plan["action_id"],"Planner selected different action")
                _equal(native["physical_budget"],plan["physical_budget"],"Remaining money/request budget")
                branches+=len(ref["branches"]);count+=1
    _require(predictions==receipts+1,"Each real reveal must refresh its actual posterior")
    return {"candidate_values_checked":count,"model_only_branches_checked":branches,
            "actual_predictions_checked":predictions,"actual_receipts_checked":receipts,"public_evidence_boundary_equal":True}


def verify(research,output):
    research,output=Path(research).resolve(),Path(output).resolve()
    root=Path(__file__).resolve().parents[1]
    _require(_hash(root/"docs/research/acquisition/round15-design.md")==DESIGN_SHA256,"Frozen mechanism design changed")
    report=json.loads((research/"report.json").read_text(encoding="utf-8"))
    _require(report["status"]=="OK" and report["distinct_return_tuples"]==40,"Research finite suite incomplete")
    for name in ("episodes.jsonl","trajectories.jsonl"):
        _require(_hash(research/name)==report["artifact_sha256"][name],"Research trace/episode hash differs")
    episodes={}
    for line in (research/"episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row["model_name"]=="correct" and row["policy"]=="adaptive-two-step":
            key=row["scenario_id"],row["world_id"]
            _require(key not in episodes,"Duplicate research episode");episodes[key]=row
    traces={}
    with (research/"trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            if row["model_name"]=="correct" and row["policy"]=="adaptive-two-step":
                key=row["scenario_id"],row["world_id"]
                _require(key not in traces,"Duplicate research trajectory");traces[key]=row["events"]
    output.mkdir(parents=True,exist_ok=False);(output/"decision-cards").mkdir()
    checks=[]
    for sid,expected_count in COUNTS.items():
        spec_ref=next(r for r in report["scenarios"] if r["scenario_id"]==sid)
        model_ref=next(r for r in report["models"] if r["scenario_id"]==sid and r["model_name"]=="correct")
        spec_path,model_path=research/spec_ref["path"],research/model_ref["path"]
        _require(_hash(spec_path)==spec_ref["sha256"] and _hash(model_path)==model_ref["sha256"],"Scenario/model file differs")
        scenario=json.loads(spec_path.read_text(encoding="utf-8"))
        patterns=select_patterns(scenario)
        _require(len(patterns)==expected_count,"Declared positive return pattern count differs")
        for returns,world_ids in patterns:
            # world IDs/outcomes remain confined to this offline verifier/scorer.
            case_id=str(uuid4())
            config={"schema":VERSION,"mode":MODE,"case_id":case_id,"prediction_cutoff":CUTOFF,
                    **{k:deepcopy(scenario[k]) for k in ("scenario_id","actions","budget_money","request_budget","lambda_money")},
                    "horizon":2,"model_version":model_ref["version"],"model_path":os.path.relpath(model_path,output),
                    "scenario_spec_path":os.path.relpath(spec_path,output),"model_sha256":_hash(model_path),"scenario_spec_sha256":_hash(spec_path)}
            config_path=output/(case_id+".json")
            config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            environment=FiniteObservationEnvironment(categories=scenario["models"]["correct"]["categories"],returns=returns)
            backend=make_finite_backend(config_path,environment)
            workflow=FiniteInformationWorkflow(backend,output/"runs")
            started=workflow.start(case_id,CUTOFF)
            final=workflow.run_to_completion(started["run_id"])
            _require(final["state"]=="STOPPED","Finite mechanism workflow did not terminate successfully")
            actions=[h["action_id"] for h in final["history"]]
            reference=episodes[sid,world_ids[0]]
            events=traces[sid,world_ids[0]]
            for ident in world_ids:
                same=episodes[sid,ident]
                _equal({k:same[k] for k in ("actions","observations","cost","final_probabilities")},
                       {k:reference[k] for k in ("actions","observations","cost","final_probabilities")},"Same return tuple across all true outcomes")
                _equal(traces[sid,ident],events,"Policy trace does not depend on realized outcome")
            _require(actions==reference["actions"],"Actual finite query sequence differs")
            _equal([final["prediction"]["probabilities"][k] for k in CLASSES],reference["final_probabilities"],"Final synthetic probability")
            _equal(final["budget"]["simulated_cost"],reference["cost"]["money"],"Actual simulated cost")
            _equal(final["budget"]["actions"],reference["cost"]["requests"],"Actual query count")
            parity=compare_run(final,events)
            audit=workflow.store.verify(final["run_id"])
            _require(audit["status"]=="OK","Durable event/blob audit failed")
            cards=build_finite_cards(final)
            card_path=output/"decision-cards"/(case_id+".json")
            card_path.write_text(json.dumps(cards,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            _require(cards["artifact_hashes"]=={"configuration":_hash(config_path),"model":_hash(model_path),"scenario_spec":_hash(spec_path)},"Card/config/model/spec binding differs")
            checks.append({"scenario_id":sid,"case_id":case_id,"run_id":final["run_id"],"actions":actions,
                           "reason":final["reason"],"simulated_cost":final["budget"]["simulated_cost"],"audit_events":audit["events_verified"],
                           "offline_reference_only":{"observations":returns,"all_matching_world_ids":world_ids,
                               "scope":"never supplied to backend prediction, estimation or acquisition"},
                           "decision_card":{"path":str(card_path.relative_to(output)),"sha256":_hash(card_path),"audit_head_hash":cards["audit_head_hash"]},
                           **parity})
        print(sid+": "+str(expected_count)+" complete positive-mass patterns verified",flush=True)
    _require(len(checks)==40 and Counter(c["scenario_id"] for c in checks)==COUNTS,"Full forty-pattern coverage failed")
    sources=("football_agent/finite_information_bridge.py","football_agent/verify_round15_finite.py",
             "football_agent/acquisition.py","football_agent/query_decision_card.py","football_agent/audit.py",
             "football_research/acquisition/finite_information.py","football_research/acquisition/joint_planner.py")
    acceptance={"status":"ROUND15_FINITE_INFORMATION_40_PATTERN_AGENT_VERIFIED","checks":checks,"pattern_counts":COUNTS,
                "design_sha256":DESIGN_SHA256,"reference_report_sha256":_hash(research/"report.json"),
                "reference_episodes_sha256":_hash(research/"episodes.jsonl"),"reference_trajectories_sha256":_hash(research/"trajectories.jsonl"),
                "implementation_sha256":{name:_hash(root/name) for name in sources},
                "new_independent_football_matches":0,"external_requests":0,"llm_calls":0,
                "scope":"Complete constructed return patterns, not independent sports samples; exact population scoring belongs to research/oracle, no confidence interval."}
    (output/"acceptance.json").write_text(json.dumps(acceptance,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return acceptance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();result=verify(args.research,args.output)
    print(json.dumps({"status":result["status"],"patterns":len(result["checks"])}))


if __name__=="__main__":main()
