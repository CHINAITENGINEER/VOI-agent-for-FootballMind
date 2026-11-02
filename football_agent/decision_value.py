"""Exact decision-focused acquisition for a finite, explicitly assumed model.

This is a small reference solver, not a learned football forecasting model.
Its observation laws and priors are inputs, never inferred from a match result.
Every tool is one-shot. Tool observations are conditionally independent given
the *complete* latent hypothesis; shared reports must be included in that
hypothesis if two tools reuse the same information.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
from types import MappingProxyType
from typing import Mapping, Sequence


PROBABILITY_TOLERANCE = 1e-10
VALUE_TOLERANCE = 1e-12
EVIDENCE_CLASS = "SYNTHETIC_ASSUMPTIONS"


def _number(value: float, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _distribution(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a probability vector")
    try:
        result = tuple(_number(v, name) for v in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a probability vector") from exc
    if len(result) != size or any(v > 1 for v in result):
        raise ValueError(f"{name} has invalid probability values or dimensions")
    if abs(math.fsum(result) - 1) > PROBABILITY_TOLERANCE:
        raise ValueError(f"{name} must sum to one")
    return result


def _names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    result = tuple(names)
    if not result or any(not isinstance(n, str) or not n.strip() for n in result):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def entropy(probabilities: Sequence[float]) -> float:
    """Natural-log entropy; inputs are validated by the public model API."""
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0)


@dataclass(frozen=True)
class ObservationTool:
    outcomes: tuple[str, ...]
    likelihoods: Mapping[str, tuple[float, ...]]
    cost: float
    description: str = ""
    evidence_id: str | None = None


@dataclass(frozen=True)
class BeliefState:
    """Posterior and observed tool responses; contains no eventual match label."""
    belief: tuple[float, ...]
    observations: tuple[tuple[str, str], ...] = ()


class FiniteDecisionModel:
    """A finite Bayes-risk model with exact one/two-step observation integration.

    P(Y, Z_1, ..., Z_n | theta) = P(Y | theta) product P(Z_i | theta).
    Correlations marginal over theta are fully retained. Unrepresented residual
    correlation violates this model; independence is not assumed merely because
    two sources have different names.

    All costs and budgets use the selected loss's units. No money-to-loss
    conversion or nominal numerical "percentage cost" is silently applied.
    """

    VERSION = "finite-conditional-evsi-v1"

    def __init__(
        self,
        hypotheses: Mapping[str, Sequence[float]],
        prior: Mapping[str, float],
        tools: Mapping[str, ObservationTool],
        *,
        outcome_labels: Sequence[str] = ("home", "draw", "away"),
        loss: str = "log_loss",
        cost_unit: str | None = None,
        assumptions: Sequence[str] = (),
    ):
        if not isinstance(hypotheses, Mapping) or not isinstance(prior, Mapping) or not isinstance(tools, Mapping):
            raise ValueError("Hypotheses, prior and tools must be mappings")
        self.hypothesis_names = _names(tuple(hypotheses), "Hypotheses")
        self.outcome_labels = _names(outcome_labels, "Outcome labels")
        if set(prior) != set(self.hypothesis_names):
            raise ValueError("Prior keys must exactly match hypotheses")
        self.prior = _distribution(tuple(prior[t] for t in self.hypothesis_names), len(self.hypothesis_names), "Prior")
        self.outcome_probabilities = tuple(
            _distribution(hypotheses[t], len(self.outcome_labels), f"P(Y|{t})") for t in self.hypothesis_names
        )
        if loss not in ("log_loss", "brier"):
            raise ValueError("Supported loss functions are log_loss and brier")
        expected_unit = "expected_log_loss_nats" if loss == "log_loss" else "expected_multiclass_brier"
        if cost_unit is not None and cost_unit != expected_unit:
            raise ValueError(f"The cost unit for {loss} must be {expected_unit}")
        self.loss, self.cost_unit = loss, expected_unit
        checked = {}
        for action_id, tool in tools.items():
            if not isinstance(action_id, str) or not action_id.strip() or action_id == "STOP":
                raise ValueError("Tool IDs must be nonempty and must not equal STOP")
            if not isinstance(tool, ObservationTool):
                raise ValueError("Tools must be ObservationTool objects")
            outcomes = _names(tool.outcomes, f"{action_id} outcomes")
            if not isinstance(tool.likelihoods, Mapping) or set(tool.likelihoods) != set(self.hypothesis_names):
                raise ValueError(f"{action_id} likelihoods must exactly cover hypotheses")
            probabilities = MappingProxyType({
                t: _distribution(tool.likelihoods[t], len(outcomes), f"P({action_id}|{t})")
                for t in self.hypothesis_names
            })
            if tool.evidence_id is not None and (not isinstance(tool.evidence_id, str) or not tool.evidence_id.strip()):
                raise ValueError("Evidence IDs must be nonempty strings when supplied")
            for other in checked.values():
                if tool.evidence_id is not None and tool.evidence_id == other.evidence_id:
                    if outcomes != other.outcomes or dict(probabilities) != dict(other.likelihoods):
                        raise ValueError("The same evidence ID must refer to the same observation law")
            checked[action_id] = ObservationTool(outcomes, probabilities, _number(tool.cost, f"{action_id} cost"), tool.description, tool.evidence_id)
        self.tools = MappingProxyType(checked)
        if isinstance(assumptions, (str, bytes)) or any(not isinstance(s, str) for s in assumptions):
            raise ValueError("Assumptions must be a sequence of strings")
        self.assumptions = tuple(assumptions)

    def initial_state(self) -> BeliefState:
        return BeliefState(self.prior)

    def _validate_state(self, state: BeliefState) -> None:
        if not isinstance(state, BeliefState):
            raise ValueError("Expected a BeliefState")
        _distribution(state.belief, len(self.hypothesis_names), "Posterior")
        seen, seen_evidence = set(), set()
        expected_belief = self.prior
        for pair in state.observations:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Observation history entries must be action/outcome pairs")
            action_id, outcome = pair
            if action_id not in self.tools or outcome not in self.tools[action_id].outcomes:
                raise ValueError("Observation history contains an unknown action/outcome")
            if action_id in seen:
                raise ValueError("Observation history repeats a one-shot tool")
            seen.add(action_id)
            evidence_id = self.tools[action_id].evidence_id
            if evidence_id is not None and evidence_id in seen_evidence:
                raise ValueError("Observation history repeats the same evidence ID")
            seen_evidence.add(evidence_id)
            j = self.tools[action_id].outcomes.index(outcome)
            weights = tuple(b * self.tools[action_id].likelihoods[t][j]
                            for b, t in zip(expected_belief, self.hypothesis_names))
            probability = math.fsum(weights)
            if probability <= 0:
                raise ValueError("Observation history is impossible under this model")
            expected_belief = tuple(w / probability for w in weights)
        if any(abs(a - b) > PROBABILITY_TOLERANCE for a, b in zip(expected_belief, state.belief)):
            raise ValueError("Posterior must match the declared prior and acquired observations")

    def prediction(self, state: BeliefState) -> tuple[float, ...]:
        self._validate_state(state)
        return tuple(math.fsum(weight * row[j] for weight, row in zip(state.belief, self.outcome_probabilities))
                     for j in range(len(self.outcome_labels)))

    def bayes_risk(self, state: BeliefState) -> float:
        predicted = self.prediction(state)
        return entropy(predicted) if self.loss == "log_loss" else 1 - math.fsum(p * p for p in predicted)

    def _available_tool(self, state: BeliefState, action_id: str) -> ObservationTool:
        self._validate_state(state)
        if not isinstance(action_id, str) or action_id not in self.tools:
            raise ValueError(f"Unknown tool: {action_id}")
        if any(action_id == old for old, _ in state.observations):
            raise ValueError(f"Tool is one-shot and was already observed: {action_id}")
        evidence_id = self.tools[action_id].evidence_id
        if evidence_id is not None and any(evidence_id == self.tools[old].evidence_id for old, _ in state.observations):
            raise ValueError(f"Evidence was already observed through another tool: {evidence_id}")
        return self.tools[action_id]

    def observation_distribution(self, state: BeliefState, action_id: str) -> dict[str, float]:
        tool = self._available_tool(state, action_id)
        return {z: math.fsum(b * tool.likelihoods[t][j] for b, t in zip(state.belief, self.hypothesis_names))
                for j, z in enumerate(tool.outcomes)}

    def observe(self, state: BeliefState, action_id: str, outcome: str) -> BeliefState:
        """Bayes update using an acquired response, never the match outcome."""
        tool = self._available_tool(state, action_id)
        if outcome not in tool.outcomes:
            raise ValueError(f"Unknown observation for {action_id}: {outcome}")
        j = tool.outcomes.index(outcome)
        weights = tuple(b * tool.likelihoods[t][j] for b, t in zip(state.belief, self.hypothesis_names))
        probability = math.fsum(weights)
        if probability <= 0:
            raise ValueError(f"Impossible observation under the current model: {action_id}/{outcome}")
        return BeliefState(tuple(w / probability for w in weights), state.observations + ((action_id, outcome),))

    def evaluate(self, state: BeliefState, *, remaining_budget: float, horizon: int = 1) -> dict:
        """Exact finite Bellman benchmark, including STOP at every branch.

        horizon=1 is myopic EVSI minus cost; horizon=2 integrates the optimal
        second action separately for each possible first observation. Missing
        observations are integrated, not filled with a most likely response.
        """
        self._validate_state(state)
        budget = _number(remaining_budget, "Remaining budget")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (0, 1, 2):
            raise ValueError("The reference solver supports integer horizons 0, 1 or 2")
        current_risk = self.bayes_risk(state)
        best_risk, chosen = current_risk, "STOP"
        candidates = []
        used = {a for a, _ in state.observations}
        used_evidence = {self.tools[a].evidence_id for a in used if self.tools[a].evidence_id is not None}
        if horizon:
            for action_id, tool in sorted(self.tools.items()):
                if action_id in used:
                    candidates.append({"action_id": action_id, "feasible": False, "reason": "already_observed"})
                    continue
                if tool.evidence_id is not None and tool.evidence_id in used_evidence:
                    candidates.append({"action_id": action_id, "feasible": False, "reason": "evidence_already_observed"})
                    continue
                if tool.cost > budget + VALUE_TOLERANCE:
                    candidates.append({"action_id": action_id, "feasible": False, "reason": "budget_exhausted"})
                    continue
                expected_risk_after = 0.0
                expected_optimal_risk_after = 0.0
                branches = []
                for outcome, probability in self.observation_distribution(state, action_id).items():
                    if probability <= 0:
                        branches.append({"observation": outcome, "probability": 0.0, "status": "impossible"})
                        continue
                    posterior = self.observe(state, action_id, outcome)
                    terminal_risk = self.bayes_risk(posterior)
                    continuation = self.evaluate(posterior, remaining_budget=max(0.0, budget - tool.cost), horizon=horizon - 1)
                    expected_risk_after += probability * terminal_risk
                    expected_optimal_risk_after += probability * continuation["expected_total_risk"]
                    branches.append({"observation": outcome, "probability": probability,
                                     "posterior": dict(zip(self.hypothesis_names, posterior.belief)),
                                     "prediction": list(self.prediction(posterior)),
                                     "terminal_risk": terminal_risk,
                                     "continuation_action": continuation["action_id"],
                                     "continuation_expected_risk": continuation["expected_total_risk"]})
                evsi = current_risk - expected_risk_after
                if evsi < -PROBABILITY_TOLERANCE:
                    raise ArithmeticError("Negative exact EVSI violates the specified Bayes-risk model")
                evsi = max(0.0, evsi)
                total_risk = tool.cost + expected_optimal_risk_after
                net_value = current_risk - total_risk
                candidates.append({"action_id": action_id, "feasible": True,
                                   "expected_risk_after_observation": expected_risk_after,
                                   "immediate_evsi": evsi, "cost": tool.cost,
                                   "immediate_net_value": evsi - tool.cost,
                                   "expected_total_risk": total_risk,
                                   "planning_net_value": net_value, "branches": branches})
                if total_risk < best_risk - VALUE_TOLERANCE:
                    best_risk, chosen = total_risk, action_id
        feasible = [c for c in candidates if c["feasible"]]
        reason = "positive_expected_net_value"
        if chosen == "STOP":
            reason = "horizon_exhausted" if horizon == 0 else (
                "no_feasible_actions" if not feasible else "no_positive_expected_net_value")
        return {"schema_version": self.VERSION, "evidence_class": EVIDENCE_CLASS,
                "action_id": chosen, "reason": reason, "horizon": horizon,
                "prediction": list(self.prediction(state)), "current_risk": current_risk,
                "expected_total_risk": best_risk, "planning_net_value": max(0.0, current_risk - best_risk),
                "remaining_budget": budget, "cost_unit": self.cost_unit, "candidates": candidates}

    def target_mutual_information(self, state: BeliefState, action_id: str) -> float:
        """I(Y; Z | currently observed evidence), independently from entropy diff."""
        tool = self._available_tool(state, action_id)
        py = self.prediction(state)
        pz = self.observation_distribution(state, action_id)
        terms = []
        for y in range(len(py)):
            for z, name in enumerate(tool.outcomes):
                joint = math.fsum(
                    b * self.outcome_probabilities[t][y] * tool.likelihoods[theta][z]
                    for t, (theta, b) in enumerate(zip(self.hypothesis_names, state.belief))
                )
                if joint > 0:
                    terms.append(joint * (math.log(joint) - math.log(py[y]) - math.log(pz[name])))
        return max(0.0, math.fsum(terms))

    def manifest(self) -> dict:
        return {"schema_version": self.VERSION, "evidence_class": EVIDENCE_CLASS,
                "hypotheses": {t: list(p) for t, p in zip(self.hypothesis_names, self.outcome_probabilities)},
                "prior": dict(zip(self.hypothesis_names, self.prior)),
                "outcome_labels": list(self.outcome_labels), "loss": self.loss, "cost_unit": self.cost_unit,
                "tools": {a: {"outcomes": list(t.outcomes), "likelihoods": {k: list(v) for k, v in t.likelihoods.items()},
                              "cost": t.cost, "description": t.description, "evidence_id": t.evidence_id} for a, t in self.tools.items()},
                "assumptions": list(self.assumptions),
                "factorization": "P(Y,Z_1,...,Z_n|theta)=P(Y|theta)*product_i P(Z_i|theta)",
                "limitations": ["Manually specified priors and likelihoods; not fitted to football observations.",
                                "Exact EVSI is conditional on the supplied model being correct.",
                                "One-shot stationary tools; no unmodeled residual source correlation.",
                                "Observed match results are evaluator-only and never policy inputs."]}

    @classmethod
    def from_manifest(cls, manifest: Mapping) -> "FiniteDecisionModel":
        if manifest.get("schema_version") != cls.VERSION or manifest.get("evidence_class") != EVIDENCE_CLASS:
            raise ValueError("Unexpected finite model schema or evidence class")
        return cls(manifest["hypotheses"], manifest["prior"],
                   {a: ObservationTool(tuple(t["outcomes"]), t["likelihoods"], t["cost"], t.get("description", ""), t.get("evidence_id"))
                    for a, t in manifest["tools"].items()},
                   outcome_labels=manifest["outcome_labels"], loss=manifest["loss"],
                   cost_unit=manifest["cost_unit"], assumptions=manifest["assumptions"])


def _draw(distribution: Sequence[float], rng: random.Random) -> int:
    threshold, cumulative = rng.random(), 0.0
    for i, probability in enumerate(distribution):
        cumulative += probability
        if threshold < cumulative:
            return i
    return len(distribution) - 1  # Floating-point sum roundoff only.


def simulate_episode(model: FiniteDecisionModel, *, seed: int, budget: float, horizon: int = 1, max_actions: int = 2) -> dict:
    """Owner samples hidden theta. Policy receives only posterior observations."""
    if model.loss not in {"log_loss", "brier"}:
        raise ValueError("Prediction simulation requires log_loss or brier; terminal decisions need their declared loss evaluator")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 0:
        raise ValueError("max_actions must be a nonnegative integer")
    remaining = _number(budget, "Budget")
    # Validate horizon even when max_actions=0.
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (0, 1, 2):
        raise ValueError("The reference solver supports integer horizons 0, 1 or 2")
    # Separate fixed streams keep potential tool responses and the match result
    # aligned when comparing horizons; policy never sees those owner variables.
    latent_index = _draw(model.prior, random.Random(f"{seed}:latent"))
    latent = model.hypothesis_names[latent_index]
    state = model.initial_state()
    trace = []
    for step in range(max_actions + 1):
        available_horizon = min(horizon, max_actions - step)
        decision = model.evaluate(state, remaining_budget=remaining, horizon=available_horizon)
        event = {"step": step, "belief_before": dict(zip(model.hypothesis_names, state.belief)),
                 "observed_before": [list(pair) for pair in state.observations], "decision": decision}
        action_id = decision["action_id"]
        if action_id == "STOP":
            trace.append(event)
            break
        tool = model.tools[action_id]
        outcome = tool.outcomes[_draw(tool.likelihoods[latent], random.Random(f"{seed}:tool:{tool.evidence_id or action_id}"))]
        state = model.observe(state, action_id, outcome)
        remaining = max(0.0, remaining - tool.cost)
        event.update({"observation": outcome, "belief_after": dict(zip(model.hypothesis_names, state.belief)),
                      "prediction_after": list(model.prediction(state)),
                      "observed_risk_after": model.bayes_risk(state), "cost": tool.cost,
                      "remaining_budget_after": remaining})
        trace.append(event)
    prediction = model.prediction(state)
    # This is sampled only after the complete policy trace. It never updates it.
    result_index = _draw(model.outcome_probabilities[latent_index], random.Random(f"{seed}:result"))
    result_loss = (-math.log(prediction[result_index]) if model.loss == "log_loss" else
                   math.fsum((p - int(i == result_index)) ** 2 for i, p in enumerate(prediction)))
    return {"schema_version": model.VERSION, "evidence_class": EVIDENCE_CLASS, "seed": seed,
            "horizon": horizon, "max_actions": max_actions, "budget": budget,
            "cost_unit": model.cost_unit, "trace": trace, "final_prediction": list(prediction),
            "spent": budget - remaining,
            "evaluator_only": {"sampled_hypothesis": latent, "sampled_result": model.outcome_labels[result_index],
                               "realized_loss": result_loss,
                               "interpretation": "One constructed realization does not estimate expected policy benefit."}}


def synthetic_scenarios() -> dict[str, dict]:
    """Predeclared, hand-constructed mechanisms; no named real match is asserted."""
    cases = {}
    base_assumption = "Every probability and cost is manually specified before simulated observations/results."

    def add(name, hypotheses, prior, tools, explanation, *, context, budget=.08):
        model = FiniteDecisionModel(hypotheses, prior, tools, assumptions=(base_assumption, explanation))
        cases[name] = {"model": model, "description": explanation, "context": context, "budget": budget,
                       "evidence_class": EVIDENCE_CLASS}

    names = tuple(f"{rotation}_{quality}" for rotation, quality in product(("starters", "reserves"), ("strong", "weak")))
    probabilities = {"starters_strong": (.75, .17, .08), "starters_weak": (.75, .17, .08),
                     "reserves_strong": (.70, .20, .10), "reserves_weak": (.35, .30, .35)}
    lineup = ObservationTool(("starters", "reserves"), {t: (1.0, 0.0) if t.startswith("starters") else (0.0, 1.0) for t in names}, .015,
                             "An assumed exact lineup observation; real announcements may have errors/delays.")
    reserves = ObservationTool(("strong", "weak"), {t: (1.0, 0.0) if t.endswith("strong") else (0.0, 1.0) for t in names}, .02,
                               "A constructed reserve-strength measurement, not a real available API.")
    add("qualified_rotation_unknown", probabilities, {t: .25 for t in names}, {"lineup": lineup, "reserve_quality": reserves},
        "Qualification is known, while rotation and reserve strength remain unknown. Rotation is not automatically weakness.",
        context={"qualification": "already_qualified", "rotation": "unknown", "reserve_strength": "unknown"})
    add("qualified_reserves_known_strong", probabilities,
        {"starters_strong": .5, "starters_weak": 0.0, "reserves_strong": .5, "reserves_weak": 0.0},
        {"lineup": lineup, "reserve_quality": reserves},
        "The same possible rotation is less informative after independent evidence has established adequate reserve strength.",
        context={"qualification": "already_qualified", "rotation": "unknown", "reserve_strength": "known_strong"})
    add("draw_sufficient_tactics_unknown", {"defensive": (.28, .55, .17), "attacking": (.55, .20, .25)},
        {"defensive": .65, "attacking": .35},
        {"tactical_evidence": ObservationTool(("defensive_signal", "attacking_signal"),
                                             {"defensive": (.85, .15), "attacking": (.15, .85)}, .01)},
        "A draw is sufficient under supplied tournament rules. Defensive play is a hypothesis, not an observed consequence; .65 is a stated prior assumption.",
        context={"qualification_condition": "draw_sufficient", "observed_tactics": "unknown"})
    add("irrelevant_high_entropy_news", {"news_a": (.45, .30, .25), "news_b": (.45, .30, .25)},
        {"news_a": .5, "news_b": .5},
        {"news": ObservationTool(("a", "b"), {"news_a": (1, 0), "news_b": (0, 1)}, .001)},
        "The news resolves one full bit of latent uncertainty but changes no outcome probability, so target EVSI is zero.",
        context={"news_status": "unknown", "news_target_relevance": "none_under_model"})
    shared = ObservationTool(("positive", "negative"), {"shared_positive": (1, 0), "shared_negative": (0, 1)}, .005,
                             "Two sources quote the same latent report; the report variable is shared, not independent evidence.")
    add("duplicate_shared_report", {"shared_positive": (.70, .20, .10), "shared_negative": (.30, .30, .40)},
        {"shared_positive": .5, "shared_negative": .5}, {"report_a": shared, "report_b": shared},
        "Both tools deterministically expose the same shared latent report. Conditional value of the duplicate becomes zero.",
        context={"sources": "same_underlying_report"})
    xor_names = tuple(f"{a}{b}" for a, b in product((0, 1), repeat=2))
    add("complementary_xor_reference", {t: (.90, .05, .05) if t[0] == t[1] else (.05, .05, .90) for t in xor_names},
        {t: .25 for t in xor_names},
        {"context_a": ObservationTool(("0", "1"), {t: (1, 0) if t[0] == "0" else (0, 1) for t in xor_names}, .02),
         "context_b": ObservationTool(("0", "1"), {t: (1, 0) if t[1] == "0" else (0, 1) for t in xor_names}, .02)},
        "A deliberately constructed XOR interaction demonstrates that two individually worthless observations can jointly improve prediction. It is a solver test, not a football synergy claim.",
        context={"interaction": "constructed_xor"}, budget=.04)
    return cases
