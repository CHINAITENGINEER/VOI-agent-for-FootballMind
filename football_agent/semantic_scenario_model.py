"""Finite, assumption-conditional VoI for a frozen semantic scenario proposal.

The proposal supplies hypotheses and observation laws, *not* action values.
The wrapper aligns its outcome marginal to an external baseline, integrates
unseen returns exactly, and replays real observations in each predeclared law.
This is not statistical calibration, nor an open-world stopping certificate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math

from .decision_value import FiniteDecisionModel, ObservationTool, VALUE_TOLERANCE
from .robust_decision_value import RobustFiniteDecisionModel
from .directional_evidence import DIRECTIONS


VERSION = "semantic-scenario-model-v1"
EVIDENCE_BASIS = "pretrained_semantic_model_assumptions_with_real_observations"
OUTCOMES = ("supports", "refutes", "no_answer")
STOP_SCOPE = "declared_model_set_and_visible_menu"
BRANCH_RISK_DIAGNOSTICS_VERSION = "branch-risk-diagnostics-v1"
BRANCH_RISK_TOLERANCE = 1e-12
PLANNING_POLICY_VERSION = "shared_observation_continuation_v1"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _distribution(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} probabilities")
    result = tuple(_number(item, name, positive=True) for item in value)
    total = math.fsum(result)
    if any(item >= 1 for item in result) or abs(total - 1) > 1e-10:
        raise ValueError(f"{name} must be strictly positive and sum to one")
    normalized = tuple(item / total for item in result)
    if any(not 0 < item < 1 for item in normalized):
        raise ValueError(f"{name} cannot be normalized with strict numerical positivity")
    return normalized


def _align_outcomes(rows, prior, baseline):
    """I-project prior[t]*rows[t,y] onto fixed theta/Y marginal constraints.

    Alternating column and row scaling preserves the common outcome logit
    shift family q(y|theta) proportional to r(y|theta)*exp(offset[y]).
    Its KL objective is sum_theta prior[theta] KL(q_theta || r_theta).
    """
    aligned = [list(row) for row in rows]
    offsets = [0.0, 0.0, 0.0]
    for iteration in range(20001):
        marginal = [math.fsum(weight * row[y] for weight, row in zip(prior, aligned))
                    for y in range(3)]
        error = max(abs(a - b) for a, b in zip(marginal, baseline))
        if error <= 1e-12:
            objective = math.fsum(
                weight * math.fsum(q * (math.log(q) - math.log(p)) for q, p in zip(new, old))
                for weight, new, old in zip(prior, aligned, rows)
            )
            return tuple(tuple(row) for row in aligned), {
                "method": "fixed_prior_joint_I_projection_common_outcome_logit_shift",
                "max_marginal_error": error,
                "iterations": iteration,
                "prior_weighted_conditional_kl": max(0.0, objective),
                "common_logit_offsets": [offset - offsets[-1] for offset in offsets],
                "statistically_calibrated": False,
            }
        multipliers = [baseline[y] / marginal[y] for y in range(3)]
        offsets = [old + math.log(scale) for old, scale in zip(offsets, multipliers)]
        for index, row in enumerate(aligned):
            scaled = [p * multiplier for p, multiplier in zip(row, multipliers)]
            total = math.fsum(scaled)
            if total <= 0 or not math.isfinite(total):
                raise ValueError("baseline alignment failed numerically")
            aligned[index] = [p / total for p in scaled]
            if any(p <= 0 for p in aligned[index]):
                raise ValueError("baseline alignment lost strict positivity")
    raise ValueError("baseline alignment did not converge to the required marginal tolerance")


class SemanticScenarioModel:
    """Bounded robust acquisition under explicit frozen semantic assumptions.

    ``planning_horizon=1`` keeps the myopic policy.  ``2`` prices a shared
    continuation: after each first observation, every declared quality model
    is evaluated with the same branchwise robust second action. A first query
    may therefore have positive planning value despite negative immediate
    value. This is a feasible bounded policy, not the globally optimal robust
    policy tree or a learned open-world transition model. Frozen manifests
    without the policy marker retain the historical reference-solver path.
    """

    VERSION = VERSION

    def __init__(self, proposal, baseline, *, request_loss_cost=.001,
                 contamination_levels=(0., .25, .5), action_request_costs=None,
                 observation_group_policy="single_informative_return",
                 emit_branch_risk_diagnostics=True, planning_horizon=1,
                 planning_policy=PLANNING_POLICY_VERSION, _legacy_output=False):
        if (not isinstance(proposal, Mapping) or proposal.get("schema") not in {
                "semantic-scenario-proposal-v1",
                "semantic-directional-scenario-proposal-v1",
        }):
            raise ValueError("unexpected semantic scenario proposal schema")
        directional_proposal = proposal.get("schema") == "semantic-directional-scenario-proposal-v1"
        hypotheses = proposal.get("hypotheses")
        actions = proposal.get("actions")
        if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 6:
            raise ValueError("proposal must have between 2 and 6 hypotheses")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 32:
            raise ValueError("proposal must have between 1 and 32 accumulated actions")
        if any(not isinstance(row, Mapping) for row in hypotheses + actions):
            raise ValueError("hypotheses and actions must contain objects")
        assumptions = proposal.get("assumptions")
        if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip()
                                                   for item in assumptions):
            raise ValueError("assumptions must be a list of nonempty strings")
        self.baseline = _distribution(baseline, 3, "baseline")
        self.request_loss_cost = _number(request_loss_cost, "request_loss_cost")
        if observation_group_policy not in ("first_attempt", "single_informative_return"):
            raise ValueError("unknown observation_group_policy")
        self.observation_group_policy = observation_group_policy
        if (isinstance(planning_horizon, bool) or not isinstance(planning_horizon, int)
                or planning_horizon not in (1, 2)):
            raise ValueError("planning_horizon must be 1 or 2")
        self.planning_horizon = planning_horizon
        if planning_policy not in (None, PLANNING_POLICY_VERSION):
            raise ValueError("unknown planning_policy")
        self.planning_policy = planning_policy
        self._legacy_output = bool(_legacy_output)
        if not isinstance(emit_branch_risk_diagnostics, bool):
            raise ValueError("emit_branch_risk_diagnostics must be a boolean")
        # Frozen runs created before this extension have no marker in their
        # manifest.  They must keep the exact old decision payload for digest
        # based math replay; newly constructed models opt in by default.
        self.emit_branch_risk_diagnostics = emit_branch_risk_diagnostics
        if not isinstance(contamination_levels, (tuple, list)) or not contamination_levels:
            raise ValueError("contamination_levels must be a nonempty sequence")
        self.contamination_levels = tuple(_number(level, "contamination level")
                                          for level in contamination_levels)
        if any(level > 1 for level in self.contamination_levels) or 0.0 not in self.contamination_levels:
            raise ValueError("contamination levels must be in [0,1] and include nominal zero")
        if len(set(self.contamination_levels)) != len(self.contamination_levels):
            raise ValueError("contamination levels must be unique")
        self.hypothesis_ids = tuple(_text(row.get("id"), "hypothesis id") for row in hypotheses)
        if len(set(self.hypothesis_ids)) != len(self.hypothesis_ids):
            raise ValueError("hypothesis IDs must be unique")
        for row in hypotheses:
            _text(row.get("description"), "hypothesis description")
        self.prior = _distribution([row.get("prior") for row in hypotheses], len(hypotheses), "prior")
        rows = tuple(_distribution(row.get("outcome_probabilities"), 3, "outcome_probabilities")
                     for row in hypotheses)
        self.aligned_outcome_probabilities, self.alignment = _align_outcomes(rows, self.prior, self.baseline)
        self.actions = {}
        self.likelihoods = {}
        action_groups = []
        for action in actions:
            action_id = _text(action.get("action_id"), "action_id")
            if action_id == "STOP" or action_id in self.actions:
                raise ValueError("action IDs must be unique and not STOP")
            question = _text(action.get("question"), "question")
            group = _text(action.get("observation_group"), "observation_group")
            action_groups.append(group)
            if directional_proposal:
                contract = action.get("direction_contract")
                if not isinstance(contract, Mapping):
                    raise ValueError("directional proposal action requires direction_contract")
                supports, refutes = contract.get("supports"), contract.get("refutes")
                if (not isinstance(supports, list) or not isinstance(refutes, list)
                        or not supports or not refutes
                        or len(supports) != len(set(supports))
                        or len(refutes) != len(set(refutes))
                        or set(supports) & set(refutes)
                        or not (set(supports) | set(refutes)) <= set(DIRECTIONS)):
                    raise ValueError("directional proposal action has invalid direction_contract")
            supplied = action.get("likelihoods")
            if not isinstance(supplied, list) or len(supplied) != len(self.hypothesis_ids):
                raise ValueError("action likelihoods must exactly cover hypotheses")
            if any(not isinstance(row, Mapping) for row in supplied):
                raise ValueError("action likelihoods must contain objects")
            law = {}
            for row in supplied:
                theta = _text(row.get("hypothesis_id"), "likelihood hypothesis_id")
                if theta in law or theta not in self.hypothesis_ids:
                    raise ValueError("action likelihoods must uniquely cover hypotheses")
                law[theta] = _distribution(row.get("probabilities"), 3, "observation probabilities")
            no_answer = law[self.hypothesis_ids[0]][2]
            if any(abs(values[2] - no_answer) > 1e-12 for values in law.values()):
                raise ValueError("no_answer likelihood must be equal across hypotheses")
            # Remove roundoff differences so no-answer cannot encode evidence.
            law = {theta: (p[0] * (1 - no_answer) / (p[0] + p[1]),
                           p[1] * (1 - no_answer) / (p[0] + p[1]), no_answer)
                   for theta, p in law.items()}
            self.actions[action_id] = {"action_id": action_id, "question": question,
                                       "observation_group": group}
            self.likelihoods[action_id] = law
        if (self.planning_policy is None and self.planning_horizon == 2
                and len(action_groups) != len(set(action_groups))):
            raise ValueError(
                "planning_horizon=2 requires unique observation_group values; "
                "group-aware continuation constraints are not silently approximated")
        supplied_costs = {} if action_request_costs is None else action_request_costs
        if not isinstance(supplied_costs, Mapping) or not set(supplied_costs) <= self.actions.keys():
            raise ValueError("action_request_costs must map known actions to positive integer request costs")
        self.action_request_costs = {}
        for action_id in self.actions:
            request_cost = supplied_costs.get(action_id, 1)
            if isinstance(request_cost, bool) or not isinstance(request_cost, int) or request_cost < 1:
                raise ValueError("action_request_costs values must be positive integers")
            if not math.isfinite(request_cost * self.request_loss_cost):
                raise ValueError("action loss cost must be finite")
            self.action_request_costs[action_id] = request_cost
        self.proposal = copy.deepcopy(dict(proposal))
        self.assumptions = tuple(assumptions) + (
            "Hypotheses, priors, outcome contrasts and observation laws are uncalibrated semantic model assumptions.",
            "Given the complete latent hypothesis, target and distinct observation groups are conditionally independent.",
            "No-answer is hypothesis-independent; unknown returns do not update beliefs.",
            ("Each observation group is acquired at most once; residual cross-group dependence is not modeled."
             if self.observation_group_policy == "first_attempt" else
             "Each action is attempted once; a group accepts at most one informative return. Hypothesis-independent no-answer or unknown permits another action in that group; residual cross-group dependence is not modeled."),
            "The contamination family is fixed before querying and is not an exhaustive open-world uncertainty set.",
            ("One-step log-loss VoI does not capture information requiring multi-step complementary acquisition."
             if self.planning_horizon == 1 else
             ("Two-step log-loss VoI prices one optimal continuation inside the frozen visible menu; "
              "it does not model undiscovered candidates or empirical transition probabilities."
              if self.planning_policy is None else
              "Two-step log-loss VoI prices a shared branchwise robust continuation inside the frozen visible menu; "
              "it is a feasible policy, not a globally optimal robust policy tree or empirical transition model.")),
        )
        prior_map = dict(zip(self.hypothesis_ids, self.prior))
        outcomes_map = dict(zip(self.hypothesis_ids, self.aligned_outcome_probabilities))
        self.models = {}
        self.model_levels = {}
        for level in self.contamination_levels:
            model_id = "contamination_" + format(level, ".17g")
            tools = {}
            for action_id, law in self.likelihoods.items():
                marginal = tuple(math.fsum(prior_map[theta] * law[theta][z] for theta in self.hypothesis_ids)
                                 for z in range(3))
                contaminated = {theta: tuple((1 - level) * value + level * marginal[z]
                                             for z, value in enumerate(law[theta]))
                                for theta in self.hypothesis_ids}
                tools[action_id] = ObservationTool(
                    OUTCOMES, contaminated, self.request_loss_cost * self.action_request_costs[action_id],
                    description=self.actions[action_id]["question"],
                    evidence_id=action_id,
                )
            self.models[model_id] = FiniteDecisionModel(outcomes_map, prior_map, tools,
                                                       loss="log_loss", assumptions=self.assumptions)
            self.model_levels[model_id] = level
        self.nominal_model_id = next(key for key, level in self.model_levels.items() if level == 0)
        self.robust_model = RobustFiniteDecisionModel(self.models, assumptions=self.assumptions)
        self.model_id = VERSION + ":" + hashlib.sha256(json.dumps(
            self.manifest(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()[:24]

    def manifest(self):
        """Rebuild parameters only: ordinary JSON, no implicit live data."""
        result = {"version": VERSION, "evidence_basis": EVIDENCE_BASIS,
                "proposal": copy.deepcopy(self.proposal), "baseline": list(self.baseline),
                "request_loss_cost": self.request_loss_cost,
                "action_request_costs": dict(self.action_request_costs),
                "contamination_levels": list(self.contamination_levels)}
        if not self._legacy_output:
            result["planning_horizon"] = self.planning_horizon
        if self.planning_policy is not None:
            result["planning_policy"] = self.planning_policy
        # Missing field is the original frozen first-attempt policy. Preserve
        # its manifest bytes/model ID rather than rewriting old runs in-place.
        if self.observation_group_policy != "first_attempt":
            result["observation_group_policy"] = self.observation_group_policy
        if self.emit_branch_risk_diagnostics:
            result["branch_risk_diagnostics"] = BRANCH_RISK_DIAGNOSTICS_VERSION
        return result

    @classmethod
    def from_manifest(cls, manifest):
        if manifest.get("version") != VERSION or manifest.get("evidence_basis") != EVIDENCE_BASIS:
            raise ValueError("unexpected semantic scenario model manifest")
        return cls(manifest["proposal"], manifest["baseline"],
                   request_loss_cost=manifest["request_loss_cost"],
                   contamination_levels=manifest["contamination_levels"],
                   action_request_costs=manifest.get("action_request_costs"),
                   observation_group_policy=manifest.get("observation_group_policy", "first_attempt"),
                   planning_horizon=manifest.get("planning_horizon", 1),
                   planning_policy=manifest.get("planning_policy"),
                   _legacy_output="planning_horizon" not in manifest,
                   emit_branch_risk_diagnostics=(
                       manifest.get("branch_risk_diagnostics") == BRANCH_RISK_DIAGNOSTICS_VERSION))

    @staticmethod
    def _branch_risk_diagnostics(decision, *, current_risk=None):
        """Summarize realized-risk branches without affecting action choice.

        ``decision`` is the exact one-step finite-model result.  A branch is
        marked as increasing risk only when its terminal predictive risk is
        strictly above the current risk by the numerical tolerance.  The
        probability is conditional on this declared model and observation
        law; it is not an empirical frequency or a forecast-error estimate.
        """
        if current_risk is None:
            current_risk = decision.get("current_risk")
        expected_after = decision.get("expected_risk_after_observation")
        cost = decision.get("cost")
        branches = decision.get("branches")
        if (not isinstance(current_risk, (int, float)) or isinstance(current_risk, bool)
                or not math.isfinite(current_risk) or not isinstance(expected_after, (int, float))
                or isinstance(expected_after, bool) or not math.isfinite(expected_after)
                or not isinstance(branches, list)):
            raise ValueError("finite decision lacks branch-risk diagnostic inputs")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost):
            raise ValueError("finite decision has invalid diagnostic cost")
        risk_increase_branches = []
        cost_risk_increase_branches = []
        branch_risk_increase_probability = 0.0
        cost_branch_risk_increase_probability = 0.0
        for branch in branches:
            probability = branch.get("probability")
            terminal_risk = branch.get("terminal_risk")
            outcome = branch.get("observation")
            if (not isinstance(probability, (int, float)) or isinstance(probability, bool)
                    or not math.isfinite(probability) or probability < 0):
                raise ValueError("finite decision has invalid branch probability")
            # Impossible branches are retained by the finite solver but cannot
            # contribute to a conditional probability.
            if probability <= 0:
                continue
            if (not isinstance(terminal_risk, (int, float)) or isinstance(terminal_risk, bool)
                    or not math.isfinite(terminal_risk)):
                raise ValueError("finite decision has invalid terminal risk")
            if terminal_risk > current_risk + BRANCH_RISK_TOLERANCE:
                branch_risk_increase_probability += probability
                risk_increase_branches.append(outcome)
            if terminal_risk + cost > current_risk + BRANCH_RISK_TOLERANCE:
                cost_branch_risk_increase_probability += probability
                cost_risk_increase_branches.append(outcome)
        return {
            "branch_risk_increase_probability": branch_risk_increase_probability,
            "expected_risk_change": expected_after - current_risk,
            "risk_increase_branches": risk_increase_branches,
            "cost_adjusted_branch_risk_increase_probability": cost_branch_risk_increase_probability,
            "cost_adjusted_risk_increase_branches": cost_risk_increase_branches,
            "risk_tolerance": BRANCH_RISK_TOLERANCE,
            "interpretation": "declared_model_conditional_branch_probability_not_empirical_frequency",
        }

    @classmethod
    def _robust_branch_risk_summary(cls, values):
        diagnostics = [value for value in values.values() if "branch_risk_increase_probability" in value]
        if not diagnostics:
            return None
        branch_sets = [set(value["risk_increase_branches"]) for value in diagnostics]
        cost_branch_sets = [set(value["cost_adjusted_risk_increase_branches"]) for value in diagnostics]
        return {
            "branch_risk_increase_probability_lower": min(value["branch_risk_increase_probability"] for value in diagnostics),
            "branch_risk_increase_probability_upper": max(value["branch_risk_increase_probability"] for value in diagnostics),
            "expected_risk_change_lower": min(value["expected_risk_change"] for value in diagnostics),
            "expected_risk_change_upper": max(value["expected_risk_change"] for value in diagnostics),
            "risk_increase_branches_union": sorted(set().union(*branch_sets)),
            "risk_increase_branches_intersection": sorted(set.intersection(*branch_sets)) if branch_sets else [],
            "cost_adjusted_branch_risk_increase_probability_lower": min(value["cost_adjusted_branch_risk_increase_probability"] for value in diagnostics),
            "cost_adjusted_branch_risk_increase_probability_upper": max(value["cost_adjusted_branch_risk_increase_probability"] for value in diagnostics),
            "cost_adjusted_risk_increase_branches_union": sorted(set().union(*cost_branch_sets)),
            "cost_adjusted_risk_increase_branches_intersection": sorted(set.intersection(*cost_branch_sets)) if cost_branch_sets else [],
            "interpretation": "declared_contamination_envelope_not_confidence_interval",
        }

    def _shared_continuation_decisions(self, decisions, observations, remaining_requests,
                                       attempted, blocked):
        """Price one implementable observation-contingent continuation.

        A branch chooses one action using the lower envelope of *one-step*
        values. That action is then priced under every model; model identity
        is never an input to the chosen policy. Integer request accounting and
        group/attempt restrictions are replayed through the public h1 path.
        This branchwise construction is conservative and need not maximize
        the worst-case value over every possible complete policy tree.
        """
        policies = {}
        for action_id, action in sorted(self.actions.items()):
            request_cost = self.action_request_costs[action_id]
            if (action_id in attempted or action["observation_group"] in blocked
                    or request_cost > remaining_requests):
                continue
            candidates = {
                model_id: next(row for row in decision["candidates"] if row["action_id"] == action_id)
                for model_id, decision in decisions.items()
            }
            expected_risks = {model_id: 0.0 for model_id in decisions}
            policy = []
            for outcome in OUTCOMES:
                continuation = self.evaluate(
                    [*observations, {"action_id": action_id, "outcome": outcome}],
                    remaining_requests - request_cost,
                    attempted_action_ids=attempted, blocked_groups=blocked, horizon=1)
                selected = continuation["selected_action_id"]
                selected_row = (next(row for row in continuation["candidates"]
                                     if row["action_id"] == selected) if selected else None)
                policy.append({"observation": outcome, "action_id": selected or "STOP",
                               "conditional_lower_net_value": selected_row["lower"] if selected_row else 0.0})
                for model_id, candidate in candidates.items():
                    branch = next(row for row in candidate["branches"] if row["observation"] == outcome)
                    # All semantic observation laws are strictly positive.
                    net = selected_row["model_values"][model_id]["net"] if selected_row else 0.0
                    risk = branch["terminal_risk"] - net
                    branch.update(continuation_action=selected or "STOP", continuation_expected_risk=risk)
                    expected_risks[model_id] += branch["probability"] * risk
            for model_id, candidate in candidates.items():
                total = candidate["cost"] + expected_risks[model_id]
                candidate.update(expected_total_risk=total,
                                 planning_net_value=decisions[model_id]["current_risk"] - total)
            policies[action_id] = policy
        return policies

    def evaluate(self, observations, remaining_requests, attempted_action_ids=(), blocked_groups=(),
                 *, horizon=None):
        if isinstance(remaining_requests, bool) or not isinstance(remaining_requests, int) or remaining_requests < 0:
            raise ValueError("remaining_requests must be a nonnegative integer")
        if horizon is None:
            horizon = self.planning_horizon
        if (isinstance(horizon, bool) or not isinstance(horizon, int)
                or horizon not in (1, 2)):
            raise ValueError("horizon must be 1 or 2")
        if horizon > self.planning_horizon:
            raise ValueError("horizon cannot exceed the model planning_horizon")
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
            raise ValueError("observations must be a sequence")
        if isinstance(attempted_action_ids, (str, bytes)) or isinstance(blocked_groups, (str, bytes)):
            raise ValueError("attempted actions and blocked groups must be collections")
        attempted, blocked = set(attempted_action_ids), set(blocked_groups)
        if not attempted <= self.actions.keys():
            raise ValueError("attempted_action_ids contains an unknown action")
        known_groups = {action["observation_group"] for action in self.actions.values()}
        if not blocked <= known_groups:
            raise ValueError("blocked_groups contains an unknown observation group")
        seen, informative_groups, updates, observed_records = set(), set(), [], []
        for observation in observations:
            if isinstance(observation, Mapping):
                action_id, outcome = observation.get("action_id"), observation.get("outcome")
            elif isinstance(observation, (tuple, list)) and len(observation) == 2:
                action_id, outcome = observation
            else:
                raise ValueError("observations must contain action/outcome records")
            if action_id not in self.actions or outcome not in (*OUTCOMES, "unknown"):
                raise ValueError("unknown observation action or outcome")
            group = self.actions[action_id]["observation_group"]
            if action_id in seen or group in informative_groups:
                raise ValueError("observations repeat a one-shot action or already informative observation group")
            seen.add(action_id)
            if outcome in ("supports", "refutes") or self.observation_group_policy == "first_attempt":
                informative_groups.add(group)
            if outcome != "unknown":
                updates.append((action_id, outcome))
            observed_records.append({"action_id": action_id, "outcome": outcome,
                                     "updates_belief": outcome not in ("unknown", "no_answer")})
        attempted.update(seen)
        blocked.update(informative_groups)
        if self.observation_group_policy == "first_attempt":
            blocked.update(self.actions[action_id]["observation_group"] for action_id in attempted)
        budget = remaining_requests * self.request_loss_cost
        if not math.isfinite(budget):
            raise ValueError("remaining request budget is not finite")
        shared_two_step = self.planning_policy is not None and horizon == 2
        finite_horizon = 1 if shared_two_step else horizon
        robust = self.robust_model.evaluate(remaining_budget=budget, horizon=finite_horizon, observations=updates)
        model_predictions, model_posteriors, decisions = {}, {}, {}
        for model_id, model in self.models.items():
            state = model.initial_state()
            for action_id, outcome in updates:
                state = model.observe(state, action_id, outcome)
            model_predictions[model_id] = list(model.prediction(state))
            model_posteriors[model_id] = dict(zip(self.hypothesis_ids, state.belief))
            decisions[model_id] = model.evaluate(state, remaining_budget=budget, horizon=finite_horizon)
        policies = (self._shared_continuation_decisions(
            decisions, observed_records, remaining_requests, attempted, blocked) if shared_two_step else {})
        raw_robust = {row["action_id"]: row for row in robust["candidates"]}
        for action_id in policies:
            values = [next(row for row in decision["candidates"] if row["action_id"] == action_id)
                      ["planning_net_value"] for decision in decisions.values()]
            raw_robust[action_id]["worst_case_planning_net_value"] = min(values)
        candidates = []
        for action_id, action in sorted(self.actions.items()):
            request_cost = self.action_request_costs[action_id]
            group_reason = ("observation_group_already_attempted_or_blocked"
                            if self.observation_group_policy == "first_attempt" else
                            "observation_group_already_informative_or_blocked")
            reason = ("already_attempted" if action_id in attempted else
                      group_reason if action["observation_group"] in blocked else
                      "request_budget_exhausted" if remaining_requests == 0 else
                      "insufficient_request_budget" if request_cost > remaining_requests else None)
            row = {**action, "cost": self.request_loss_cost * request_cost, "cost_unit": "expected_log_loss_nats",
                   "request_cost": request_cost, "gross": None, "net": None, "lower": None, "upper": None,
                   "gross_lower": None, "gross_upper": None,
                   "feasible": reason is None, "reason": reason}
            if reason is None:
                values = {}
                for model_id, decision in decisions.items():
                    candidate = next(item for item in decision["candidates"] if item["action_id"] == action_id)
                    if not candidate["feasible"]:
                        raise ArithmeticError("wrapper feasible action disagrees with finite solver")
                    values[model_id] = {"gross": candidate["immediate_evsi"],
                                        "net": candidate["immediate_net_value"]}
                    if not self._legacy_output:
                        values[model_id].update({
                            "planning_net": candidate["planning_net_value"],
                            "planning_gross": candidate["planning_net_value"] + candidate["cost"],
                            "continuation_actions": [branch.get("continuation_action")
                                                      for branch in candidate.get("branches", ())]})
                    if self.emit_branch_risk_diagnostics:
                        values[model_id].update(self._branch_risk_diagnostics(
                            candidate, current_risk=decision["current_risk"]))
                envelope = raw_robust[action_id]
                lower = envelope["worst_case_planning_net_value"]
                if abs(lower - min(value.get("planning_net", value["net"]) for value in values.values())) > 1e-10:
                    raise ArithmeticError("robust lower envelope disagrees with finite planning integration")
                nominal = values[self.nominal_model_id]
                immediate_lower = min(value["net"] for value in values.values())
                immediate_upper = max(value["net"] for value in values.values())
                planning_upper = max(value.get("planning_net", value["net"]) for value in values.values())
                row.update(gross=nominal["gross"], net=nominal["net"], lower=lower,
                           upper=(immediate_upper if self._legacy_output else planning_upper),
                           gross_lower=min(value["gross"] for value in values.values()),
                           gross_upper=max(value["gross"] for value in values.values()),
                           model_values=values,
                           reason="positive_declared_lower_net_value" if lower > VALUE_TOLERANCE
                                  else "nonpositive_declared_lower_net_value")
                if self.emit_branch_risk_diagnostics:
                    robust_diagnostics = self._robust_branch_risk_summary(values)
                    row.update(
                        branch_risk_increase_probability=nominal["branch_risk_increase_probability"],
                        expected_risk_change=nominal["expected_risk_change"],
                        risk_increase_branches=list(nominal["risk_increase_branches"]),
                        cost_adjusted_branch_risk_increase_probability=(
                            nominal["cost_adjusted_branch_risk_increase_probability"]),
                        cost_adjusted_risk_increase_branches=list(
                            nominal["cost_adjusted_risk_increase_branches"]),
                        risk_diagnostics={
                            "nominal": {key: value for key, value in nominal.items()
                                        if key not in ("gross", "net")},
                            "robust": robust_diagnostics,
                            "contamination_models": {
                                model_id: {key: value for key, value in model_value.items()
                                           if key not in ("gross", "net")}
                                for model_id, model_value in values.items()},
                        },
                        robust_branch_risk_diagnostics=robust_diagnostics,
                    )
                    if robust_diagnostics is not None:
                        # Flat aliases keep the audit payload easy to consume
                        # while the nested object preserves nominal/robust
                        # provenance for downstream report code.
                        row.update({
                            key: value for key, value in robust_diagnostics.items()
                            if key != "interpretation"
                        })
            candidates.append(row)
            if action_id in policies:
                row["continuation_policy"] = policies[action_id]
        feasible = [row for row in candidates if row["feasible"]]
        best = max(feasible, key=lambda row: (row["lower"], row["action_id"])) if feasible else None
        selected = best["action_id"] if best is not None and best["lower"] > VALUE_TOLERANCE else None
        reason = (None if selected is not None else "request_budget_exhausted" if remaining_requests == 0 else
                  "insufficient_request_budget" if not feasible and any(
                      row["reason"] == "insufficient_request_budget" for row in candidates) else
                  "no_unattempted_observation_groups" if not feasible else "no_positive_declared_lower_net_value")
        result = {"version": VERSION, "model_id": self.model_id, "evidence_basis": EVIDENCE_BASIS,
                "assumptions": list(self.assumptions), "statistically_calibrated": False,
                "alignment": dict(self.alignment), "baseline": list(self.baseline),
                "aligned_outcome_probabilities": {theta: list(row) for theta, row in
                                                  zip(self.hypothesis_ids, self.aligned_outcome_probabilities)},
                "prediction_nominal": model_predictions[self.nominal_model_id],
                "posterior_nominal": model_posteriors[self.nominal_model_id],
                "model_predictions": model_predictions, "model_posteriors": model_posteriors,
                "model_contamination_levels": dict(self.model_levels),
                "observations": observed_records, "candidates": candidates,
                "selected_action_id": selected, "stop_reason": reason,
                "stop_scope": STOP_SCOPE, "global_stop_certificate": False,
                "horizon": horizon,
                "remaining_requests": remaining_requests,
                "lower_upper_interpretation": "net_value_envelope_not_confidence_interval"}
        if not self._legacy_output:
            result.update({"planning_horizon": horizon,
                           "candidate_value_semantics": {
                               "immediate_net": "one_query_evsi_minus_cost",
                               "planning_net": ("shared_branchwise_robust_continuation_policy_value"
                                                if shared_two_step else "finite_bellman_value_including_one_continuation") if horizon == 2
                               else "same_as_immediate_net_when_horizon_one"},})
            for row in result["candidates"]:
                if row.get("feasible"):
                    values = row["model_values"]
                    nominal = values[self.nominal_model_id]
                    row.update({"immediate_lower": min(value["net"] for value in values.values()),
                                "immediate_upper": max(value["net"] for value in values.values()),
                                "planning_net": nominal["planning_net"],
                                "planning_gross": nominal["planning_gross"],
                                "continuation_actions": nominal["continuation_actions"],
                                "planning_lower": row["lower"],
                                "planning_upper": max(value["planning_net"] for value in values.values())})
        if self.planning_policy is not None:
            result["planning_policy"] = self.planning_policy
            result["planning_policy_scope"] = "shared_observation_contingent_actions_with_declared_model_risks"
            result["global_policy_optimality_claim"] = False
        if self.emit_branch_risk_diagnostics:
            result["branch_risk_diagnostics"] = {
                "version": BRANCH_RISK_DIAGNOSTICS_VERSION,
                "risk_tolerance": BRANCH_RISK_TOLERANCE,
                "definition": "P(terminal_risk > current_risk + tolerance | declared model and observation law)",
                "cost_adjusted_definition": "P(terminal_risk + deterministic_cost > current_risk + tolerance | declared model and observation law)",
                "interpretation": "conditional_model_probability_not_empirical_frequency",
                "robust_interpretation": "per-model diagnostics and finite envelope; not a confidence interval",
            }
        return result
