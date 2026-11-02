"""Durable synthetic information acquisition with environment-only hidden returns."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from uuid import UUID

from football_research.acquisition.joint_planner import JointRiskPlanner, _budget, _cost
from .acquisition import AcquisitionWorkflow, public_state
from .audit import digest
from .query_decision_card import build_query_decision_cards
from .types import utc


VERSION = "finite-information-agent-backend-v1"
MODE = "synthetic_finite_population"
CUTOFF = "2026-09-27T00:00:00+00:00"
CLASSES = ("home", "draw", "away")  # Legacy transport aliases only; cards use 0/1/2.


class FiniteObservationEnvironment:
    """Contains a complete return tuple, no realized outcome or world identifier."""
    def __init__(self, *, categories, returns):
        if not isinstance(returns, dict) or set(returns) != set(categories):
            raise ValueError("Exactly the declared observation tuple is required")
        if any(not isinstance(v,str) or v not in categories[k] for k,v in returns.items()):
            raise ValueError("Unsupported environment return category")
        self.__returns = deepcopy(returns)
        self.__observations, self.__history, self.__requests = {}, [], set()
        self.__lock = RLock()

    def check(self, state):
        with self.__lock:
            if state != {"observations": self.__observations, "action_history": self.__history}:
                raise ValueError("Public observation ledger differs from actually acquired evidence")

    def reveal(self, action_id, request_id):
        with self.__lock:
            if not isinstance(request_id,str) or not request_id or request_id in self.__requests:
                raise ValueError("Unique acquisition request ID required")
            if action_id not in self.__returns or action_id in self.__observations:
                raise ValueError("Unique unacquired action required")
            token = self.__returns[action_id]
            self.__observations[action_id] = token
            self.__history.append({"action_id":action_id,"status":"observed"})
            self.__requests.add(request_id)
            return token


class FiniteInformationBackend:
    """Only the environment object can reveal a current case's hidden returns."""
    def __init__(self, *, environment, model, configuration):
        self._environment, self.model = environment, model
        self._configuration = deepcopy(configuration)
        self.case_id = str(UUID(configuration["case_id"]))
        self.scenario_id = configuration["scenario_id"]
        self.cutoff = utc(configuration.get("prediction_cutoff",CUTOFF)).isoformat()
        if configuration.get("schema") != VERSION or configuration.get("mode") != MODE:
            raise ValueError("Explicit finite synthetic backend configuration required")
        if configuration.get("model_version") != model.model_version:
            raise ValueError("Finite model version differs from bound configuration")
        if configuration.get("horizon") not in (1,2) or type(configuration.get("horizon")) is not int:
            raise ValueError("Finite planning horizon must be one or two")
        self.catalog = deepcopy(configuration["actions"])
        if len({a["action_id"] for a in self.catalog}) != len(self.catalog):
            raise ValueError("Unique catalog identities required")
        for action in self.catalog:
            _cost(action)
        self.planner = JointRiskPlanner(model,horizon=configuration["horizon"],lambda_cost={"money":configuration["lambda_money"]})
        _budget({"limits":{"money":configuration["budget_money"],"requests":configuration["request_budget"]},"spent":{}})

    def configuration(self):
        return {**deepcopy(self._configuration),"backend_version":VERSION,"planner_version":self.planner.version,
                "risk_mode":"mean_only","gain_semantics":"expected_terminal_log_loss_reduction_over_conditional_plan",
                "cost_semantics":"expected_total_plan_penalty_not_current_charge",
                "class_transport_aliases":{"home":0,"draw":1,"away":2},
                "class_scope":"synthetic_categories_not_football_result_labels"}

    def _state(self, visible):
        clean = public_state(visible)
        if set(clean)!={"match_id","prediction_cutoff","information_mode","base_facts","groups","mask",
                        "action_history","evidence_versions","quality","state_version","budget"} or clean["quality"]!={"synthetic":True}:
            raise ValueError("Finite public envelope contains extra metadata or hypothetical state")
        if (clean["match_id"] != self.case_id or clean["prediction_cutoff"] != self.cutoff
                or clean.get("information_mode") != MODE or clean.get("base_facts") != {"scenario_id":self.scenario_id}):
            raise ValueError("Finite case identity, cutoff or public scenario differs")
        keys = {a["action_id"] for a in self.catalog}
        if set(clean["mask"]) != keys or set(clean["groups"]) != {k for k,v in clean["mask"].items() if v}:
            raise ValueError("Acquired groups must exactly match finite action masks")
        observations = {}
        for aid,group in clean["groups"].items():
            if set(group) != {"status","values"} or group["status"] != "observed" or set(group["values"]) != {"category"}:
                raise ValueError("Only acquired categorical tokens may enter finite groups")
            observations[aid] = group["values"]["category"]
        state = {"observations":observations,"action_history":deepcopy(clean["action_history"])}
        if clean["state_version"]!=len(observations) or clean["evidence_versions"]!={aid:self.model.model_version for aid in observations}:
            raise ValueError("Acquired finite evidence versions differ")
        self.model.observed_actions(state,hypothetical=False)
        self._environment.check(state)
        limits,spent=_budget(self._budget(clean))
        if limits["money"]>self._configuration["budget_money"]+1e-12 or limits["requests"]>self._configuration["request_budget"]:
            raise ValueError("Runtime budget exceeds declared synthetic scenario limits")
        actual_money=sum(_cost(a)["money"] for a in self.catalog if a["action_id"] in observations)
        if abs(spent["money"]-actual_money)>1e-12 or spent["requests"]!=len(observations):
            raise ValueError("Runtime spending differs from actual acquired observation ledger")
        return state

    def start(self, match_id, cutoff, seed=0):
        if type(seed) is not int:raise ValueError("Integer execution seed required; worlds are supplied by environment")
        if match_id != self.case_id or utc(cutoff).isoformat() != self.cutoff:
            raise ValueError("Configured opaque case identity and cutoff required")
        state = {"match_id":self.case_id,"prediction_cutoff":self.cutoff,"information_mode":MODE,
                 "base_facts":{"scenario_id":self.scenario_id},"groups":{},
                 "mask":{a["action_id"]:False for a in self.catalog},"action_history":[],
                 "evidence_versions":{},"quality":{"synthetic":True},"state_version":0,
                 "budget":{"max_simulated_cost":self._configuration["budget_money"],"simulated_cost":0.,
                           "max_actions":self._configuration["request_budget"],"actions":0}}
        self._state(state)
        return state

    @staticmethod
    def _budget(visible):
        raw=visible["budget"]
        return {"limits":{"money":raw["max_simulated_cost"],"requests":raw["max_actions"]},
                "spent":{"money":raw["simulated_cost"],"requests":raw["actions"]}}

    def predict(self, visible):
        state=self._state(visible)
        prediction=self.model.predict_partial(state)
        return {"match_id":self.case_id,"probabilities":dict(zip(CLASSES,prediction["probabilities"])),
                "model_version":self.model.model_version,"observed_return_categories":deepcopy(state["observations"]),
                "mask":deepcopy(visible["mask"]),"mode":MODE,"class_order":[0,1,2]}

    def actions(self, visible):
        state=self._state(visible)
        return [{"action_id":a["action_id"],"group_ids":[a["action_id"]],"parameters":{},
                 "status":"UNAVAILABLE" if a["action_id"] in state["observations"] else "AVAILABLE",
                 "reason":"ALREADY_ACQUIRED" if a["action_id"] in state["observations"] else None,
                 "cost":{"amount":a["cost"]["money"],"status":"simulated","unit":"simulation_credit"},
                 "cost_vector":_cost(a),"cost_scope":"current_synthetic_observation",
                 "version":self.model.model_version,"source":"constructed_finite_environment"} for a in self.catalog]

    def estimate(self, visible, actions):
        state=self._state(visible)
        catalog={a["action_id"]:a for a in self.catalog}
        if len({a["action_id"] for a in actions})!=len(actions) or any(a["action_id"] not in catalog for a in actions):
            raise ValueError("Unknown or duplicate finite candidate")
        plan=self.planner.plan(state,[catalog[a["action_id"]] for a in actions],self._budget(visible))
        return [{"action_id":c["action_id"],"status":"OK" if c["feasible"] else "UNAVAILABLE","reason":c["reason"],
                 "expected_loss_improvement":c.get("expected_loss_improvement"),"cost_in_loss_units":c.get("expected_plan_penalty"),
                 "net_gain":c.get("selection_score"),"lower_gain":None,"standard_error":None,
                 "uncertainty_kind":"known_constructed_joint_distribution_no_parameter_interval",
                 "estimator_version":self.planner.version,
                 "research_estimate":{**deepcopy(c),"planned_action":plan["action_id"],"plan_reason":plan["reason"],
                                      "physical_budget":plan["physical_budget"],"lambda_cost":plan["lambda_cost"]}} for c in plan["candidates"]]

    def acquire(self, visible, action_id, parameters, request_id):
        self._state(visible)
        if parameters != {}:raise ValueError("Finite categorical actions take no parameters")
        action=next((a for a in self.catalog if a["action_id"]==action_id),None)
        if action is None:raise ValueError("Unknown finite action")
        limits,spent=_budget(self._budget(visible))
        cost=_cost(action)
        if any(spent[u]+cost[u]>cap+1e-12 for u,cap in limits.items()):
            raise ValueError("Physical budget exhausted")
        token=self._environment.reveal(action_id,request_id)
        updated=deepcopy(visible)
        updated["groups"][action_id]={"status":"observed","values":{"category":token}}
        updated["mask"][action_id]=True
        updated["action_history"].append({"action_id":action_id,"status":"observed"})
        updated["evidence_versions"][action_id]=self.model.model_version
        updated["state_version"]+=1
        updated["budget"]["simulated_cost"]+=cost["money"]
        updated["budget"]["actions"]+=1
        self._state(updated)
        return {"status":"OK","action_id":action_id,"request_id":request_id,"visible_state":updated,
                "observation":token,"requests":0,
                "cost":{"amount":cost["money"],"status":"simulated","unit":"simulation_credit"},
                "cost_vector":cost,"scope":"actual_synthetic_return_not_model_branch"}


def make_finite_backend(config_path, environment):
    from football_research.acquisition.finite_information import FiniteJointModel
    path=Path(config_path).resolve()
    config=json.loads(path.read_text(encoding="utf-8"))
    for key in ("model","scenario_spec"):
        artifact=(path.parent/config[key+"_path"]).resolve()
        if sha256(artifact.read_bytes()).hexdigest()!=config[key+"_sha256"]:
            raise ValueError(key+" artifact hash mismatch")
    model=FiniteJointModel.load((path.parent/config["model_path"]).resolve())
    scenario=json.loads((path.parent/config["scenario_spec_path"]).read_text(encoding="utf-8"))
    if any(config[k]!=scenario[k] for k in ("scenario_id","actions","budget_money","request_budget","lambda_money")):
        raise ValueError("Runtime action/budget configuration differs from bound scenario specification")
    if model.spec!=scenario["models"].get(model.spec["model_name"]):
        raise ValueError("Model distribution differs from the bound scenario's declared model variant")
    config["artifact_hashes"]={"configuration":sha256(path.read_bytes()).hexdigest(),
                               "model":config["model_sha256"],"scenario_spec":config["scenario_spec_sha256"]}
    # Paths are loading inputs, not runtime tool/Agent metadata.
    config.pop("model_path");config.pop("scenario_spec_path")
    return FiniteInformationBackend(environment=environment,model=model,configuration=config)


class FiniteInformationWorkflow(AcquisitionWorkflow):
    version="durable-finite-information-workflow-v1"
    def start(self,match_id,cutoff,strategy="all_candidates",budget=None,seed=0,risk_mode="mean"):
        if strategy!="all_candidates" or risk_mode!="mean" or self.backend.configuration().get("mode")!=MODE:
            raise ValueError("Finite benchmark requires its mean numeric planner")
        if budget is None:
            cfg=self.backend.configuration()
            budget={"max_simulated_cost":cfg["budget_money"],"max_actions":cfg["request_budget"],"max_llm_calls":0}
        return super().start(match_id,cutoff,strategy=strategy,budget=budget,seed=seed,risk_mode=risk_mode)


def build_finite_cards(run):
    cards=build_query_decision_cards(run)
    events={e["seq"]:e for e in run["events"]}
    cfg=run["configuration"]["backend"]
    cards.update(scenario_id=cfg["scenario_id"],mode=MODE,
                 class_semantics="known_synthetic_class_0_1_2_not_football",tool_scope="declared_finite_categorical_queries",
                 scope="Synthetic mechanism verification; not independent football evidence or LLM behavior.")
    def relabel(context):
        if context and context.get("probabilities") is not None:
            context["probabilities"]={str(i):context["probabilities"][name] for i,name in enumerate(CLASSES)}
        return context
    cards["final_probabilities"]={str(i):cards["final_probabilities"][name] for i,name in enumerate(CLASSES)}
    for card in cards["cards"]:
        relabel(card["current"])
        event=events[card["audit"]["seq"]]
        native={a["action_id"]:a.get("value",{}).get("research_estimate",{}) for a in event["data"].get("candidates",[])}
        card["current"]["acquired_tokens"]={aid:g["values"]["category"] for aid,g in event["snapshot"]["visible_state"]["groups"].items()}
        for candidate in card["candidates"]:
            raw=native[candidate["action_id"]]
            candidate.update(plan_value=deepcopy(raw),value_scope="total_conditional_plan_gain_and_penalty",
                             branch_scope="hypothetical_model_categories_not_actual_receipts")
        if card.get("receipt"):
            receipt_event=events[card["receipt"]["audit"]["seq"]]
            card["receipt"]["observation"]=receipt_event["data"]["result"]["observation"]
            card["receipt"]["return_kind"]="actual_synthetic_categorical_return"
            relabel(card["receipt"]["prediction_after_return"])
    return cards
