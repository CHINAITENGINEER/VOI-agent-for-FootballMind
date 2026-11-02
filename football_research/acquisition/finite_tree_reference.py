"""Independent finite-policy enumeration for the round15 mechanism benchmark.

No production predictor, planner, generator or scoring helper is imported.
Every policy is a first action and a Cartesian product of branch-wise second
actions/STOP. Expectations are direct weighted sums over explicitly given rows.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from itertools import combinations, permutations, product
import math

STRATEGIES = ("base-only", "myopic", "adaptive-two-step", "best-fixed-subset", "best-fixed-order-stop")


def entropy(values):
    return -math.fsum(v * math.log(v) for v in values if v > 0)


def frozen_population(name):
    """Generate the predeclared true law independently of the production code."""
    rows = []
    if name in {"gated_stop", "gated_route"}:
        costs = {"context": .02, "signal_a": .20}
        if name == "gated_route":
            costs["signal_b"] = .20
        for y, g, a in product(range(3), range(2), range(3)):
            ka = (.9 if a == y else .05) if g == 0 else 1/3
            if name == "gated_stop":
                rows.append({"mass": ka/6, "outcome": y,
                             "observations": {"context": str(g), "signal_a": str(a)}})
            else:
                for b in range(3):
                    kb = (.9 if b == y else .05) if g == 1 else 1/3
                    rows.append({"mass": ka*kb/6, "outcome": y,
                                 "observations": {"context": str(g), "signal_a": str(a), "signal_b": str(b)}})
        cap = .22 if name == "gated_stop" else .42
    elif name == "complementary":
        costs, cap = {"signal_a": .12, "signal_b": .12}, .24
        for a, b, y in product(range(3), repeat=3):
            rows.append({"mass": (.9 if y == (a+b) % 3 else .05)/9, "outcome": y,
                         "observations": {"signal_a": str(a), "signal_b": str(b)}})
    elif name == "redundant":
        costs, cap = {"signal_a": .04, "signal_b": .06}, .10
        for y, a in product(range(3), repeat=2):
            rows.append({"mass": (.9 if y == a else .05)/3, "outcome": y,
                         "observations": {"signal_a": str(a), "signal_b": str(a)}})
    elif name == "irrelevant":
        costs, cap = {"signal_a": .02, "signal_b": .03}, .05
        for y, a, b in product(range(3), range(2), range(2)):
            rows.append({"mass": (.7,.2,.1)[y] * (.6,.4)[a] * (.3,.7)[b], "outcome": y,
                         "observations": {"signal_a": str(a), "signal_b": str(b)}})
    else:
        raise ValueError("Unknown frozen scenario")
    actions = [{"action_id": a, "cost_status": "simulated", "cost": {"money": c, "requests": 1}}
               for a, c in sorted(costs.items())]
    return rows, actions, cap


def distorted_population(rows, name):
    if name == "correct":
        return deepcopy(rows)
    if name == "swapped_context":
        return [{**deepcopy(r), "observations": {**r["observations"], "context": str(1-int(r["observations"]["context"]))}}
                for r in rows]
    if name != "independent":
        raise ValueError("Unknown frozen model variant")
    names = sorted(rows[0]["observations"])
    categories = {a: sorted({r["observations"][a] for r in rows}) for a in names}
    prior = [math.fsum(r["mass"] for r in rows if r["outcome"] == y) for y in range(3)]
    marginals = {a: {z: math.fsum(r["mass"] for r in rows if r["observations"][a] == z)
                     for z in categories[a]} for a in names}
    output = []
    for tokens in product(*(categories[a] for a in names)):
        observations = dict(zip(names, tokens))
        for y in range(3):
            output.append({"mass": prior[y] * math.prod(marginals[a][z] for a, z in observations.items()),
                           "outcome": y, "observations": observations})
    return output


class FiniteTreeReference:
    def __init__(self, rows, actions, money_cap, requests=2, weight=1.):
        self.rows = deepcopy(rows)
        self.costs = {a["action_id"]: float(a["cost"]["money"]) for a in actions}
        self.cap, self.requests, self.weight = float(money_cap), requests, float(weight)
        if len(self.costs) != len(actions) or requests not in (0,1,2):
            raise ValueError("Unique actions and at most two requests required")
        if any(a["cost"]["requests"] != 1 for a in actions):
            raise ValueError("Reference expects one request per query")
        if any(not math.isfinite(c) or c < 0 for c in [self.cap, self.weight, *self.costs.values()]):
            raise ValueError("Invalid physical cost")
        if not rows or abs(math.fsum(r["mass"] for r in rows)-1.) > 1e-10:
            raise ValueError("Normalized positive finite population required")
        for row in rows:
            if (type(row["mass"]) not in (float,int) or not math.isfinite(row["mass"]) or row["mass"] <= 0
                    or type(row["outcome"]) is not int or row["outcome"] not in range(3)
                    or set(row["observations"]) != set(self.costs)
                    or any(not isinstance(z,str) or not z for z in row["observations"].values())):
                raise ValueError("Invalid finite population row")

    @staticmethod
    def _key(observations):
        return tuple(sorted(observations.items()))

    @lru_cache(None)
    def conditioned(self, key):
        obs = dict(key)
        if set(obs)-set(self.costs):
            raise ValueError("Unknown observed action")
        subset = [r for r in self.rows if all(r["observations"][a] == z for a,z in obs.items())]
        mass = math.fsum(r["mass"] for r in subset)
        if mass <= 0:
            raise ValueError("Unsupported observation")
        prior = tuple(math.fsum(r["mass"] for r in subset if r["outcome"] == y)/mass for y in range(3))
        return subset, mass, prior

    def posterior(self, observations):
        return self.conditioned(self._key(observations))[2]

    def branches(self, observations, action):
        if action not in self.costs or action in observations:
            raise ValueError("Unknown or repeated query")
        rows, mass, _ = self.conditioned(self._key(observations))
        return [(z, math.fsum(r["mass"] for r in rows if r["observations"][action] == z)/mass)
                for z in sorted({r["observations"][action] for r in rows})]

    def legal(self, observations):
        spent = math.fsum(self.costs[a] for a in observations)
        if len(observations) >= self.requests:
            return []
        return [a for a in sorted(self.costs) if a not in observations and spent+self.costs[a] <= self.cap+1e-12]

    def _tree(self, observations, action, children):
        risk = entropy(self.posterior(observations))
        if action is None:
            return {"action_id": None, "children": {}, "expected_terminal_risk": risk,
                    "expected_money": 0., "expected_queries": 0., "expected_total_risk": risk,
                    "net_gain": 0., "selection_score": 0.}
        leaves, fees, calls = [], [], []
        for token, probability in self.branches(observations, action):
            first = {**observations, action: token}
            second = children[token]
            if second is None:
                terminal = entropy(self.posterior(first))
                fee, count = self.costs[action], 1.
            else:
                terminal = math.fsum(w*entropy(self.posterior({**first, second:z})) for z,w in self.branches(first, second))
                fee, count = self.costs[action]+self.costs[second], 2.
            leaves.append(probability*terminal)
            fees.append(probability*fee)
            calls.append(probability*count)
        terminal, money, queries = math.fsum(leaves), math.fsum(fees), math.fsum(calls)
        total = terminal + self.weight*money
        return {"action_id": action, "children": dict(children), "expected_terminal_risk": terminal,
                "expected_money": money, "expected_queries": queries, "expected_total_risk": total,
                "net_gain": risk-total, "selection_score": round(risk-total,12)}

    def trees(self, observations=None, horizon=2):
        observations = observations or {}
        if type(horizon) is not int or horizon not in (1,2):
            raise ValueError("One or two step enumeration required")
        yield self._tree(observations, None, {})
        for action in self.legal(observations):
            branches = self.branches(observations,action)
            options = []
            for z,_ in branches:
                second = self.legal({**observations,action:z}) if horizon == 2 else []
                options.append([None, *second])
            for decisions in product(*options):
                yield self._tree(observations, action, dict(zip((z for z,_ in branches), decisions)))

    @lru_cache(None)
    def _solve(self, key, horizon):
        best = None
        by_action, count = {}, 0
        for tree in self.trees(dict(key), horizon):
            count += 1
            aid = tree["action_id"]
            if aid not in by_action or tree["expected_total_risk"] < by_action[aid]["expected_total_risk"]-1e-12:
                by_action[aid] = tree
        for aid in [None,*sorted(a for a in by_action if a is not None)]:
            tree = by_action[aid]
            if best is None or tree["selection_score"] > best["selection_score"]:
                best = tree
        return {**best, "candidate_optima": {a:t for a,t in by_action.items() if a is not None},
                "enumerated_policy_trees": count}

    def solve(self, observations=None, horizon=2):
        return deepcopy(self._solve(self._key(observations or {}), horizon))

    @lru_cache(None)
    def fixed_subset(self):
        choices = []
        for size in range(self.requests+1):
            for order in combinations(sorted(self.costs),size):
                fee = math.fsum(self.costs[a] for a in order)
                if fee > self.cap+1e-12:
                    continue
                groups = {}
                for row in self.rows:
                    key = tuple((a,row["observations"][a]) for a in order)
                    groups.setdefault(key,[]).append(row["mass"])
                terminal = math.fsum(math.fsum(mass)*entropy(self.posterior(dict(key))) for key,mass in groups.items())
                choices.append({"order": list(order), "expected_terminal_risk": terminal,
                                "expected_money": fee, "expected_total_risk": terminal+self.weight*fee})
        return min(choices,key=lambda r:(round(r["expected_total_risk"],12),len(r["order"]),r["order"]))

    @lru_cache(None)
    def fixed_order_stop(self):
        choices = [{**self._tree({},None,{}), "order": []}]
        for size in range(1,self.requests+1):
            for order in permutations(sorted(self.costs),size):
                if math.fsum(self.costs[a] for a in order) > self.cap+1e-12:
                    continue
                first = order[0]
                children = {}
                for z,_ in self.branches({},first):
                    obs = {first:z}
                    second = order[1] if size == 2 else None
                    gain = (entropy(self.posterior(obs))-math.fsum(w*entropy(self.posterior({**obs,second:t}))
                            for t,w in self.branches(obs,second))-self.weight*self.costs[second]) if second else 0.
                    children[z] = second if round(gain,12)>0 else None
                tree = self._tree({},first,children)
                if tree["selection_score"] <= 0:
                    tree = self._tree({},None,{})
                choices.append({**tree,"order":list(order)})
        return min(choices,key=lambda r:(round(r["expected_total_risk"],12),len(r["order"]),r["order"]))

    def action(self, strategy, observations):
        if strategy == "base-only":
            return None
        if strategy in {"myopic","adaptive-two-step"}:
            return self.solve(observations, 1 if strategy == "myopic" else 2)["action_id"]
        if strategy == "best-fixed-subset":
            order = self.fixed_subset()["order"]
            return next((a for a in order if a not in observations),None)
        if strategy == "best-fixed-order-stop":
            tree = self.fixed_order_stop()
            if not observations:
                return tree["action_id"]
            if len(observations) >= 2 or tree["action_id"] not in observations:
                return None
            return tree["children"].get(observations[tree["action_id"]])
        raise ValueError("Unknown policy")

    def path(self, strategy, returns):
        observations, events, actions = {}, [], []
        while True:
            prediction = list(self.posterior(observations))
            action = self.action(strategy, observations)
            events.append({"observations": dict(observations), "probabilities": prediction, "action_id": action})
            if action is None:
                break
            if action not in self.legal(observations):
                raise AssertionError("Reference policy exceeded a physical budget")
            observations[action] = returns[action]
            actions.append(action)
        return {"events": events, "actions": actions, "observations": observations,
                "probabilities": list(self.posterior(observations)),
                "money": math.fsum(self.costs[a] for a in actions)}

    def evaluate(self, strategy, population=None):
        population = self.rows if population is None else population
        terms, paths = [], []
        for row in population:
            trace = self.path(strategy,row["observations"])
            p = trace["probabilities"][row["outcome"]]
            if p <= 0:
                raise ValueError("A scored true outcome has zero modeled probability")
            loss = -math.log(p)
            terms.append((row["mass"],loss,trace["money"],len(trace["actions"])))
            paths.append({"mass":row["mass"],"actions":trace["actions"]})
        return {"log_loss":math.fsum(w*l for w,l,_,_ in terms),
                "money":math.fsum(w*c for w,_,c,_ in terms),
                "queries":math.fsum(w*q for w,_,_,q in terms),
                "utility_loss":math.fsum(w*(l+self.weight*c) for w,l,c,_ in terms)}
