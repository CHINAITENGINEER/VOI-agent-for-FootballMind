"""Predeclared finite information worlds and a public-evidence joint adapter."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

from .joint_planner import entropy
from .value import _canonical

VERSION = "finite-information-model-v1"
SCENARIO_VERSION = "finite-information-scenario-v1"
HYPOTHESIS_KEY = "joint_return_hypothesis"
FROZEN_TIME = "2026-09-27T00:00:00+00:00"


def _k(z,y):
    return .9 if z==y else .05


def _action(aid,cost):
    return {"action_id":aid,"available":True,"cost_status":"simulated","evidence_id":"finite:"+aid,
        "cost":{"money":cost,"requests":1.,"latency_ms":0.,"tokens":0.}}


def _model_spec(scenario,name,action_ids,rows):
    return {"schema":VERSION,"scenario_id":scenario,"model_name":name,"action_ids":list(action_ids),
        "categories":{aid:sorted({r["observations"][aid] for r in rows}) for aid in action_ids},
        "rows":deepcopy(rows),"metadata":{"training_as_of":FROZEN_TIME,"model_origin":"no_fit_declared_distribution",
            "timestamp_semantics":"frozen specification declaration; no learned fitting or historical availability claim",
            "scenario_id":scenario,"model_name":name,"information_mode":"synthetic_enumerated_population"}}


def build_scenarios():
    definitions={
        "gated_stop":([_action("context",.02),_action("signal_a",.20)],.22),
        "gated_route":([_action("context",.02),_action("signal_a",.20),_action("signal_b",.20)],.42),
        "complementary":([_action("signal_a",.12),_action("signal_b",.12)],.24),
        "redundant":([_action("signal_a",.04),_action("signal_b",.06)],.10),
        "irrelevant":([_action("signal_a",.02),_action("signal_b",.03)],.05)}
    output={}
    for sid,(actions,cap) in definitions.items():
        rows=[]
        if sid=="gated_stop":
            for y,g,a in product(range(3),range(2),range(3)):
                rows.append({"mass":(1/3)*.5*(_k(a,y) if g==0 else 1/3),"outcome":y,"observations":{"context":str(g),"signal_a":str(a)}})
        elif sid=="gated_route":
            for y,g,a,b in product(range(3),range(2),range(3),range(3)):
                rows.append({"mass":(1/3)*.5*(_k(a,y) if g==0 else 1/3)*(_k(b,y) if g==1 else 1/3),
                    "outcome":y,"observations":{"context":str(g),"signal_a":str(a),"signal_b":str(b)}})
        elif sid=="complementary":
            for y,a,b in product(range(3),repeat=3):
                rows.append({"mass":(1/9)*_k(y,(a+b)%3),"outcome":y,"observations":{"signal_a":str(a),"signal_b":str(b)}})
        elif sid=="redundant":
            for y,a in product(range(3),repeat=2):
                rows.append({"mass":(1/3)*_k(a,y),"outcome":y,"observations":{"signal_a":str(a),"signal_b":str(a)}})
        else:
            for y,a,b in product(range(3),range(2),range(2)):
                rows.append({"mass":(.7,.2,.1)[y]*(.6,.4)[a]*(.3,.7)[b],"outcome":y,"observations":{"signal_a":str(a),"signal_b":str(b)}})
        aids=[a["action_id"] for a in actions]
        models={"correct":_model_spec(sid,"correct",aids,rows)}
        py={y:math.fsum(r["mass"] for r in rows if r["outcome"]==y) for y in range(3)}
        categories=models["correct"]["categories"]
        marginals={aid:{z:math.fsum(r["mass"] for r in rows if r["observations"][aid]==z) for z in categories[aid]} for aid in aids}
        independent=[]
        for y in range(3):
            for values in product(*(categories[a] for a in aids)):
                obs=dict(zip(aids,values))
                independent.append({"mass":py[y]*math.prod(marginals[a][obs[a]] for a in aids),"outcome":y,"observations":obs})
        models["independent"]=_model_spec(sid,"independent",aids,independent)
        if sid.startswith("gated_"):
            swapped=deepcopy(rows)
            for row in swapped:row["observations"]["context"]=str(1-int(row["observations"]["context"]))
            models["swapped_context"]=_model_spec(sid,"swapped_context",aids,swapped)
        output[sid]={"schema":SCENARIO_VERSION,"scenario_id":sid,"actions":actions,"budget_money":cap,"request_budget":2,"lambda_money":1.,
            "worlds":[{"world_id":f"{sid}:{i:04d}",**deepcopy(r)} for i,r in enumerate(rows) if r["mass"]>0],"models":models}
    return output


class FiniteJointModel:
    def __init__(self,spec):
        if not isinstance(spec,dict) or spec.get("schema")!=VERSION:
            raise ValueError("Finite information model schema required")
        required={"schema","scenario_id","model_name","action_ids","categories","rows","metadata"}
        if set(spec)!=required:raise ValueError("Unexpected or missing finite model fields")
        aids=spec["action_ids"]
        if not isinstance(aids,list) or not aids or any(not isinstance(a,str) or not a for a in aids) or len(set(aids))!=len(aids):
            raise ValueError("Unique nonempty action identities required")
        categories=spec["categories"]
        if not isinstance(categories,dict) or set(categories)!=set(aids):raise ValueError("Exact category action keys required")
        for values in categories.values():
            if not isinstance(values,list) or not values or any(not isinstance(z,str) or not z for z in values) or len(set(values))!=len(values):
                raise ValueError("Unique categorical observation strings required")
        rows=spec["rows"]
        if not isinstance(rows,list) or not rows:raise ValueError("Nonempty finite distribution required")
        seen=set()
        for row in rows:
            if not isinstance(row,dict) or set(row)!={"mass","outcome","observations"}:raise ValueError("Exact distribution row fields required")
            m,y,obs=row["mass"],row["outcome"],row["observations"]
            if isinstance(m,bool) or not isinstance(m,(int,float)) or not math.isfinite(m) or m<=0:raise ValueError("Strictly positive finite distribution mass required")
            if type(y) is not int or y not in (0,1,2):raise ValueError("Outcome index must be 0, 1 or 2")
            if not isinstance(obs,dict) or set(obs)!=set(aids) or any(not isinstance(obs[a],str) or obs[a] not in categories[a] for a in aids):
                raise ValueError("Complete modeled categorical observations required")
            key=(y,tuple(obs[a] for a in aids))
            if key in seen:raise ValueError("Duplicate outcome/observation world")
            seen.add(key)
        if abs(math.fsum(r["mass"] for r in rows)-1)>1e-10:raise ValueError("Distribution mass must sum to one")
        if any(set(categories[a])!={r["observations"][a] for r in rows} for a in aids):raise ValueError("Category without positive support")
        if not isinstance(spec["metadata"],dict) or not isinstance(spec["metadata"].get("training_as_of"),str):raise ValueError("Declared specification metadata required")
        for key in ("scenario_id","model_name"):
            if not isinstance(spec[key],str) or not spec[key] or spec["metadata"].get(key)!=spec[key]:raise ValueError("Scenario/model metadata identity mismatch")
        self.spec=deepcopy(spec);self.action_ids=tuple(aids);self.categories=deepcopy(categories);self.metadata=deepcopy(spec["metadata"])
        self.model_version=self.version=VERSION+":"+sha256(_canonical(self.spec).encode()).hexdigest()[:20]
        self._cache={}

    def _state(self,state,*,hypothetical=False):
        allowed={"observations","action_history"}|({HYPOTHESIS_KEY} if hypothetical else set())
        if not isinstance(state,dict) or set(state)-allowed or "observations" not in state:
            raise ValueError("Only acquired observations and public action history are allowed at runtime")
        observations=state["observations"]
        if not isinstance(observations,dict):raise ValueError("Observations must be a mapping")
        def check(obs):
            if not isinstance(obs,dict) or set(obs)-set(self.action_ids):raise ValueError("Unknown observation action")
            if any(not isinstance(z,str) or z not in self.categories[a] for a,z in obs.items()):raise ValueError("Unsupported observation category")
        check(observations)
        history=state.get("action_history",[])
        if not isinstance(history,list):raise ValueError("Public history must be a list")
        attempted=set()
        for event in history:
            if not isinstance(event,dict) or set(event)!={"action_id","status"} or event["action_id"] not in observations:
                raise ValueError("History must refer to acquired evidence only")
            if event["status"] not in {"observed","acquired"}:raise ValueError("Unsupported or failed acquisition status")
            if event["action_id"] in attempted:raise ValueError("Duplicate acquisition in history")
            attempted.add(event["action_id"])
        result=dict(observations)
        if HYPOTHESIS_KEY in state:
            envelope=state[HYPOTHESIS_KEY]
            if (not isinstance(envelope,dict) or set(envelope)!={"schema","hypothetical","observations"}
                    or envelope.get("schema")!=VERSION or envelope.get("hypothetical") is not True):
                raise ValueError("Invalid explicit planning-only observation envelope")
            check(envelope["observations"])
            for aid,z in envelope["observations"].items():
                if aid in result and result[aid]!=z:raise ValueError("Hypothetical observation conflicts with acquired evidence")
                result[aid]=z
        return result

    def _distribution(self,observations):
        key=tuple(sorted(observations.items()))
        if key not in self._cache:
            rows=[r for r in self.spec["rows"] if all(r["observations"][a]==z for a,z in observations.items())]
            mass=math.fsum(r["mass"] for r in rows)
            if mass<=0:raise ValueError("Observation combination has zero modeled support")
            p=[math.fsum(r["mass"] for r in rows if r["outcome"]==y)/mass for y in range(3)]
            self._cache[key]=(rows,mass,p)
        return self._cache[key]

    def _predict(self,state,*,hypothetical):
        observations=self._state(state,hypothetical=hypothetical)
        _,mass,p=self._distribution(observations)
        return {"probabilities":list(p),"model_version":self.model_version,"observations":observations,"evidence_probability":mass,
            "prediction_kind":"model_only_hypothetical" if hypothetical else "acquired_evidence_posterior"}

    def predict_partial(self,state):return self._predict(state,hypothetical=False)
    def predict_hypothetical(self,state):return self._predict(state,hypothetical=True)

    def observed_actions(self,state,*,hypothetical=False):
        obs=self._state(state,hypothetical=hypothetical);self._distribution(obs)
        return sorted(obs)

    def enumerate_action(self,state,action,*,hypothetical=False):
        observations=self._state(state,hypothetical=hypothetical)
        aid=action.get("action_id") if isinstance(action,dict) else None
        if aid not in self.action_ids:raise ValueError("Unmodeled action identity")
        if aid in observations:raise ValueError("One-shot tool already acquired")
        rows,total,p=self._distribution(observations)
        branches=[]
        for z in self.categories[aid]:
            mass=math.fsum(r["mass"] for r in rows if r["observations"][aid]==z)
            if mass<=0:continue
            after={**observations,aid:z};_,_,posterior=self._distribution(after)
            # Keep actual receipts at the root; merged hypothetical evidence is
            # in the planning envelope only and never promoted into receipts.
            nxt={"observations":deepcopy(state["observations"]),"action_history":deepcopy(state.get("action_history",[])),
                HYPOTHESIS_KEY:{"schema":VERSION,"hypothetical":True,"observations":after}}
            branches.append({"observation":z,"probability":mass/total,"posterior_probabilities":list(posterior),
                "next_state":nxt,"return_status":"observed"})
        expected=math.fsum(b["probability"]*entropy(b["posterior_probabilities"]) for b in branches)
        return {"action_id":aid,"current_probabilities":list(p),"branches":branches,"expected_posterior_entropy":expected,
            "evsi":max(0.,entropy(p)-expected),"scope":"exact declared finite model, not established real-world information value"}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"spec":self.spec,"model_version":self.model_version},ensure_ascii=False,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        if set(payload)!={"spec","model_version"}:raise ValueError("Finite model artifact fields mismatch")
        model=cls(payload["spec"])
        if model.model_version!=payload["model_version"]:raise ValueError("Finite model version/hash mismatch")
        return model
