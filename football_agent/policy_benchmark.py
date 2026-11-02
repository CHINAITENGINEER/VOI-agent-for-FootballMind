"""Pure offline policy replay and metrics for Round133-C."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations, permutations
import math
import random
from typing import Any

from .value_evaluation_contract import INDEPENDENT_GOLD as INDEPENDENT_GOLD

VERSION = "policy-benchmark-v1"
POLICIES = ("fixed_all", "fixed_subset", "first_query", "least_cost",
            "current_voi_gate", "stop_immediately", "random",
            "oracle_upper_bound")
_EPS = 1e-12
_MAX_ORACLE_CANDIDATES = 10
_MAX_ORACLE_COMBINATIONS = 4096


def _finite(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum - _EPS:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum + _EPS:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _sidecar(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("evaluation_sidecar", "evaluator_sidecar",
                "independent_evaluation", "evaluation"):
        value = episode.get(key)
        if isinstance(value, Mapping):
            return value
    raise ValueError("independent evaluator sidecar is required")


def _public(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    value = episode.get("public")
    return value if isinstance(value, Mapping) else episode


def _same(values: Sequence[tuple[str, Any]], name: str) -> Any:
    values = [(path, value) for path, value in values if value is not None]
    if not values:
        return None
    first = values[0][1]
    if any(value != first for _, value in values[1:]):
        raise ValueError(f"{name} conflict")
    return first


def _rows(sidecar: Mapping[str, Any], episode: Mapping[str, Any], *,
          evaluator_id: str, baseline: float, loss_unit: str) -> dict[str, dict[str, Any]]:
    raw = sidecar.get("candidate_evaluations", episode.get("candidate_evaluations"))
    if isinstance(raw, Mapping):
        raw = [dict(v, candidate_id=k) if isinstance(v, Mapping)
               else {"candidate_id": k, "decision_loss_after": v}
               for k, v in raw.items()]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("candidate_evaluations must be a sequence or mapping")
    result: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw):
        if not isinstance(raw_row, Mapping):
            raise ValueError("candidate evaluation rows must be objects")
        candidate_id = _text(raw_row.get("candidate_id"), f"candidate_evaluations[{index}].candidate_id")
        if candidate_id in result:
            raise ValueError(f"duplicate evaluator candidate: {candidate_id}")
        row = dict(raw_row)
        if row.get("evaluator_id") != evaluator_id:
            raise ValueError(f"{candidate_id} evaluator_id mismatch")
        if row.get("label_status") != INDEPENDENT_GOLD:
            raise ValueError(f"{candidate_id} is not independent evaluator gold")
        if row.get("loss_unit") != loss_unit:
            raise ValueError(f"{candidate_id} loss_unit mismatch")
        row_baseline = row.get("baseline_loss", row.get("decision_loss_before"))
        if row_baseline is None:
            raise ValueError(f"{candidate_id} baseline_loss is required")
        row_baseline = _finite(row_baseline, f"{candidate_id}.baseline_loss", minimum=0)
        if abs(row_baseline - baseline) > _EPS:
            raise ValueError(f"{candidate_id} baseline_loss mismatch")
        row["baseline_loss"] = row_baseline
        if "decision_loss_before" in row:
            before = _finite(row["decision_loss_before"], f"{candidate_id}.decision_loss_before", minimum=0)
            if abs(before - baseline) > _EPS:
                raise ValueError(f"{candidate_id} decision_loss_before mismatch")
        if "decision_loss_after" in row and "loss_after" in row and row["decision_loss_after"] != row["loss_after"]:
            raise ValueError(f"{candidate_id} loss_after aliases conflict")
        if "decision_loss_after" in row:
            row["decision_loss_after"] = _finite(row["decision_loss_after"], f"{candidate_id}.decision_loss_after", minimum=0)
            row["loss_after"] = row["decision_loss_after"]
        elif "loss_after" in row:
            row["loss_after"] = _finite(row["loss_after"], f"{candidate_id}.loss_after", minimum=0)
        for key, minimum, maximum in (("reliability", 0, 1), ("latency_ms", 0, None), ("predicted_risk", 0, 1)):
            if key in row and row[key] is not None:
                row[key] = _finite(row[key], f"{candidate_id}.{key}", minimum=minimum, maximum=maximum)
        for key in ("useful", "contradiction", "acceptance"):
            if key in row and row[key] is not None and not isinstance(row[key], bool):
                raise ValueError(f"{candidate_id}.{key} must be boolean")
        result[candidate_id] = row
    return result


def _menu(episode: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    public = _public(episode)
    raw = public.get("candidate_menu", public.get("candidates"))
    if raw is None:
        pre = episode.get("pre_action")
        ids = pre.get("candidate_ids", ()) if isinstance(pre, Mapping) else ()
        raw = [{"candidate_id": item} for item in ids]
    if isinstance(raw, Mapping):
        raw = [dict(v, candidate_id=k) if isinstance(v, Mapping)
               else {"candidate_id": k, "expected_value": v} for k, v in raw.items()]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("public candidate menu must be a sequence or mapping")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if isinstance(value, str):
            value = {"candidate_id": value}
        if not isinstance(value, Mapping):
            raise ValueError("public candidate menu rows must be objects")
        candidate_id = _text(value.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate public candidate: {candidate_id}")
        seen.add(candidate_id)
        item = {"candidate_id": candidate_id}
        for key, minimum, maximum in (("cost", 0, None), ("expected_value", 0, None),
                                      ("latency_ms", 0, None), ("reliability", 0, 1),
                                      ("predicted_risk", 0, 1)):
            item[key] = None if value.get(key) is None else _finite(value[key], f"{candidate_id}.{key}", minimum=minimum, maximum=maximum)
        if "acceptance" in value:
            if value["acceptance"] is not None and not isinstance(value["acceptance"], bool):
                raise ValueError(f"{candidate_id}.acceptance must be boolean")
            item["acceptance"] = value["acceptance"]
        result.append(item)
    if not result:
        raise ValueError("episode must declare at least one public candidate")
    if set(rows) - seen:
        raise ValueError("evaluator candidate extends public candidate menu")
    return tuple(result)


def _sequence_key(key: Any) -> tuple[str, ...]:
    parts = list(key) if isinstance(key, (list, tuple)) else ([] if str(key) == "" else str(key).replace("|", ">").split(">"))
    sequence = tuple(_text(item, "sequence candidate") for item in parts)
    if len(set(sequence)) != len(sequence):
        raise ValueError("sequence contains duplicate candidate")
    return sequence


def _sequence_losses(sidecar: Mapping[str, Any], menu: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]], baseline: float) -> dict[tuple[str, ...], float]:
    raw = sidecar.get("sequence_losses")
    if not isinstance(raw, Mapping):
        raise ValueError("sequence_losses must be an object")
    ids = {item["candidate_id"] for item in menu}
    result: dict[tuple[str, ...], float] = {}
    for key, value in raw.items():
        sequence = _sequence_key(key)
        if any(item not in ids for item in sequence) or sequence in result:
            raise ValueError("invalid or duplicate sequence")
        result[sequence] = _finite(value, "sequence_loss", minimum=0)
    if () not in result or abs(result[()] - baseline) > _EPS:
        raise ValueError("empty sequence baseline is required and must match baseline_loss")
    for candidate_id, row in rows.items():
        after = row.get("loss_after")
        if after is not None and (candidate_id,) in result and abs(after - result[(candidate_id,)]) > _EPS:
            raise ValueError(f"{candidate_id} singleton loss mismatch")
    return result


def normalize_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(episode, Mapping):
        raise ValueError("episode must be an object")
    episode_id = _text(episode.get("episode_id"), "episode_id")
    sidecar = _sidecar(episode)
    evaluator_block = sidecar.get("evaluator")
    evaluator_id_value = sidecar.get("evaluator_id")
    if evaluator_id_value is None and isinstance(evaluator_block, Mapping):
        evaluator_id_value = evaluator_block.get("evaluator_id")
    evaluator_id = _text(evaluator_id_value, "evaluator_id")
    label_status = sidecar.get("label_status")
    if label_status != INDEPENDENT_GOLD:
        raise ValueError("evaluator sidecar label_status must be INDEPENDENT_GOLD")
    loss_unit = _text(sidecar.get("loss_unit"), "loss_unit")
    baseline = _finite(sidecar.get("baseline_loss"), "baseline_loss", minimum=0)
    if episode.get("baseline_loss") is not None and abs(_finite(episode["baseline_loss"], "episode.baseline_loss", minimum=0) - baseline) > _EPS:
        raise ValueError("episode and sidecar baseline_loss conflict")
    rows = _rows(sidecar, episode, evaluator_id=evaluator_id, baseline=baseline, loss_unit=loss_unit)
    menu = _menu(episode, rows)
    public = _public(episode)
    budget_value = public.get("budget", episode.get("budget"))
    budget = None if budget_value is None else _finite(budget_value, "budget", minimum=0)
    cost_unit = _same((("public.cost_unit", public.get("cost_unit")), ("episode.cost_unit", episode.get("cost_unit")), ("sidecar.cost_unit", sidecar.get("cost_unit"))), "cost_unit")
    cost_unit = None if cost_unit is None else _text(cost_unit, "cost_unit")
    rate_value = _same((("public.cost_to_loss_rate", public.get("cost_to_loss_rate")), ("episode.cost_to_loss_rate", episode.get("cost_to_loss_rate")), ("sidecar.cost_to_loss_rate", sidecar.get("cost_to_loss_rate"))), "cost_to_loss_rate")
    rate = None if rate_value is None else _finite(rate_value, "cost_to_loss_rate", minimum=0)
    if (cost_unit is None) != (rate is None):
        raise ValueError("cost_unit and cost_to_loss_rate must be provided together")
    pre = episode.get("pre_action")
    complete_generation = public.get("candidate_generation_complete") is True or (isinstance(pre, Mapping) and pre.get("candidate_generation_complete") is True)
    return {"episode_id": episode_id, "budget": budget, "menu": menu, "rows": rows,
            "baseline_loss": baseline, "sequence_losses": _sequence_losses(sidecar, menu, rows, baseline),
            "evaluator_id": evaluator_id, "loss_unit": loss_unit, "cost_unit": cost_unit,
            "cost_to_loss_rate": rate, "candidate_generation_complete": complete_generation,
            "complete_sequence_evaluation": sidecar.get("complete_sequence_evaluation") is True,
            "fixed_subset": tuple(public.get("fixed_subset", ())) if isinstance(public.get("fixed_subset", ()), (list, tuple)) else (),
            "terminal_continuation_label": sidecar.get("terminal_continuation_label", sidecar.get("terminal_continuation"))}


def _loss(case: Mapping[str, Any], selected: Sequence[str]) -> float | None:
    return case["sequence_losses"].get(tuple(selected))


def _cost(case: Mapping[str, Any], selected: Sequence[str]) -> float | None:
    by_id = {item["candidate_id"]: item for item in case["menu"]}
    values = [by_id[item]["cost"] for item in selected]
    return None if any(value is None for value in values) else math.fsum(values)


def _latency(case: Mapping[str, Any], selected: Sequence[str]) -> float | None:
    if not selected:
        return None
    by_id = {item["candidate_id"]: item for item in case["menu"]}
    values = [case["rows"].get(item, {}).get("latency_ms", by_id[item].get("latency_ms")) for item in selected]
    return None if any(value is None for value in values) else math.fsum(values)


def _affordable(case: Mapping[str, Any], order: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if case["budget"] is None:
        return [], True
    selected: list[dict[str, Any]] = []
    remaining = float(case["budget"])
    deferred = False
    for item in order:
        if item.get("cost") is None:
            deferred = True
        elif item["cost"] <= remaining + _EPS:
            selected.append(item)
            remaining = max(0.0, remaining - item["cost"])
    return selected, deferred


def _fixed_all(case: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Choose a budget-feasible public-menu baseline without evaluator labels.

    When the public menu declares numeric expected values, the fixed-all
    baseline uses the highest-value feasible subset.  This keeps its resource
    spend comparable under a finite budget while remaining actor-visible.  If
    any public estimate is unknown, it falls back to stable menu order and
    leaves the uncertainty explicit through ``deferred``.
    """
    menu = list(case["menu"])
    if case["budget"] is None:
        return [], True
    if any(item.get("cost") is None or item.get("expected_value") is None for item in menu):
        return _affordable(case, menu)
    if len(menu) > 16:
        ordered = sorted(menu, key=lambda item: (-(item["expected_value"] / item["cost"]) if item["cost"] > _EPS else -math.inf, item["candidate_id"]))
        return _affordable(case, ordered)
    best: tuple[dict[str, Any], ...] = ()
    best_value = -math.inf
    best_cost = math.inf
    for size in range(len(menu) + 1):
        for subset in combinations(menu, size):
            cost = math.fsum(item["cost"] for item in subset)
            if cost > case["budget"] + _EPS:
                continue
            value = math.fsum(item["expected_value"] for item in subset)
            ids = tuple(item["candidate_id"] for item in subset)
            best_ids = tuple(item["candidate_id"] for item in best)
            if (value > best_value + _EPS or
                    (abs(value - best_value) <= _EPS and
                     (cost < best_cost - _EPS or (abs(cost - best_cost) <= _EPS and ids < best_ids)))):
                best, best_value, best_cost = subset, value, cost
    return list(best), False


def _combination_count(n: int) -> int:
    total, value = 0, 1
    for k in range(n + 1):
        if k:
            value *= n - k + 1
        total += value
    return total


def _oracle_qualified(case: Mapping[str, Any]) -> bool:
    # Bound checks must precede completeness checks and permutation traversal.
    if len(case["menu"]) > _MAX_ORACLE_CANDIDATES or _combination_count(len(case["menu"])) > _MAX_ORACLE_COMBINATIONS:
        return False
    if not case.get("candidate_generation_complete") or not case.get("complete_sequence_evaluation"):
        return False
    if case.get("cost_unit") is None or case.get("cost_to_loss_rate") is None or case.get("budget") is None:
        return False
    ids = tuple(item["candidate_id"] for item in case["menu"])
    if set(case["rows"]) != set(ids) or any(item.get("cost") is None for item in case["menu"]):
        return False
    return all(sequence in case["sequence_losses"] for length in range(len(ids) + 1) for sequence in permutations(ids, length))


def _oracle(case: Mapping[str, Any]) -> tuple[str, ...] | None:
    if not _oracle_qualified(case):
        return None
    by_id = {item["candidate_id"]: item for item in case["menu"]}
    ids = list(by_id)
    best, best_utility = (), -math.inf
    rate = float(case["cost_to_loss_rate"])
    for length in range(len(ids) + 1):
        for sequence in permutations(ids, length):
            spend = math.fsum(by_id[item]["cost"] for item in sequence)
            if spend > case["budget"] + _EPS:
                continue
            loss = _loss(case, sequence)
            if loss is None:
                continue
            utility = case["baseline_loss"] - loss - spend * rate
            if utility > best_utility + _EPS or (abs(utility - best_utility) <= _EPS and sequence < best):
                best, best_utility = tuple(sequence), utility
    return best


def _selection(case: Mapping[str, Any], policy_id: str, seed: int) -> tuple[tuple[str, ...] | None, str]:
    menu = list(case["menu"])
    if policy_id == "oracle_upper_bound":
        selected = _oracle(case)
        return selected, "oracle_selected" if selected is not None else "oracle_ineligible"
    if policy_id == "stop_immediately":
        return (), "stop_immediately"
    if case["budget"] is None:
        return (), "defer_unknown_budget"
    if policy_id == "fixed_all":
        selected, deferred = _fixed_all(case)
        return tuple(item["candidate_id"] for item in selected), "defer_unknown_cost" if deferred else "policy_exhausted"
    if policy_id == "fixed_subset":
        subset = set(case["fixed_subset"]) or {item["candidate_id"] for item in menu[:max(1, (len(menu) + 1) // 2)]}
        if not subset <= {item["candidate_id"] for item in menu}:
            raise ValueError("fixed_subset contains candidate outside public menu")
        selected, deferred = _affordable(case, [item for item in menu if item["candidate_id"] in subset])
        return tuple(item["candidate_id"] for item in selected), "defer_unknown_cost" if deferred else "policy_exhausted"
    if policy_id == "first_query":
        selected, deferred = _affordable(case, menu[:1])
        return tuple(item["candidate_id"] for item in selected), "defer_unknown_cost" if deferred else "policy_exhausted"
    if policy_id == "least_cost":
        ordered = sorted(menu, key=lambda item: (item["cost"] is None, item["cost"] if item["cost"] is not None else math.inf, item["candidate_id"]))
        selected, deferred = _affordable(case, ordered)
        return tuple(item["candidate_id"] for item in selected), "defer_unknown_cost" if deferred else "policy_exhausted"
    if policy_id == "current_voi_gate":
        remaining, selected, pending = float(case["budget"]), [], list(menu)
        while pending:
            feasible = [item for item in pending if item.get("cost") is not None and item["cost"] <= remaining + _EPS]
            if not feasible:
                return tuple(selected), "defer_unknown_estimate" if any(item.get("cost") is None for item in pending) else "policy_exhausted"
            # An unknown public value is a stopping uncertainty of its own.
            # Do not let a low-valued known candidate make the policy appear
            # decisive while a feasible unknown candidate remains.
            if any(item.get("expected_value") is None for item in feasible):
                return tuple(selected), "defer_unknown_estimate"
            item = max(feasible, key=lambda row: (-math.inf if row.get("expected_value") is None else row["expected_value"] - row["cost"], -math.inf if row.get("expected_value") is None else row["expected_value"], -row["cost"], row["candidate_id"]))
            if item.get("expected_value") is None:
                return tuple(selected), "defer_unknown_estimate"
            if item["expected_value"] <= item["cost"] + _EPS:
                break
            selected.append(item["candidate_id"])
            remaining = max(0.0, remaining - item["cost"])
            pending.remove(item)
        return tuple(selected), "policy_exhausted"
    if policy_id == "random":
        random.Random(seed).shuffle(menu)
        selected, deferred = _affordable(case, menu)
        return tuple(item["candidate_id"] for item in selected), "defer_unknown_cost" if deferred else "policy_exhausted"
    raise ValueError(f"unknown policy: {policy_id}")


def _annotation_metrics(case: Mapping[str, Any], selected: Sequence[str]) -> tuple[float | None, int | None, bool | None]:
    by_id = {item["candidate_id"]: item for item in case["menu"]}
    reliabilities, contradictions, risks, accepts = [], [], [], []
    for item in selected:
        row = case["rows"].get(item, {})
        reliabilities.append(row.get("reliability", by_id[item].get("reliability")))
        contradictions.append(row.get("contradiction"))
        risks.append(row.get("predicted_risk", by_id[item].get("predicted_risk")))
        accepts.append(row.get("acceptance", by_id[item].get("acceptance")))
    reliability = None if not selected or any(v is None for v in reliabilities) else math.fsum(reliabilities) / len(reliabilities)
    contradiction_count = None if not selected or any(v is None for v in contradictions) else sum(bool(v) for v in contradictions)
    selective = None if not selected or any(v is None for v in risks + accepts) else all(bool(a) and float(r) <= .5 for r, a in zip(risks, accepts))
    return reliability, contradiction_count, selective


def replay_policy(episode: Mapping[str, Any], policy_id: str, *, seed: int = 0, reliability_threshold: float | None = None) -> dict[str, Any]:
    if policy_id not in POLICIES:
        raise ValueError(f"unknown policy: {policy_id}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    threshold = None if reliability_threshold is None else _finite(reliability_threshold, "reliability_threshold", minimum=0, maximum=1)
    case = normalize_episode(episode)
    selected, status = _selection(case, policy_id, seed)
    if policy_id == "oracle_upper_bound" and selected is None:
        return {"schema_version": VERSION, "episode_id": case["episode_id"], "policy_id": policy_id, "seed": seed, "selected_candidates": None, "selection_status": status, "stop_reason": "oracle_ineligible", "spent_cost": None, "latency_ms": None, "budget_remaining": None, "baseline_loss": None, "final_loss": None, "decision_loss_reduction": None, "decision_loss_reduction_per_cost": None, "net_utility": None, "final_selection_regret_proxy": None, "regret_vs_oracle": None, "oracle_utility": None, "oracle_optimal": None, "candidate_precision": None, "candidate_recall": None, "positive_candidate_count": None, "stop_quality": None, "reliability_coverage": None, "contradiction_count": None, "selective_risk_covered": None, "evaluator_only": True, "runtime_safe": False}
    selected = tuple(selected or ())
    spent, latency = _cost(case, selected), _latency(case, selected)
    final_loss = _loss(case, selected)
    reduction = None if final_loss is None else case["baseline_loss"] - final_loss
    rate = case["cost_to_loss_rate"]
    converted_cost = None if spent is None or rate is None else spent * rate
    utility = None if reduction is None or converted_cost is None else reduction - converted_cost
    ratio = None if reduction is None or converted_cost is None or converted_cost <= _EPS else reduction / converted_cost
    positive = {key for key, row in case["rows"].items() if row.get("useful") is True or (row.get("useful") is None and row.get("loss_after") is not None and row["loss_after"] < case["baseline_loss"] - _EPS)}
    selected_positive = len(set(selected) & positive)
    precision = selected_positive / len(selected) if selected else None
    recall = selected_positive / len(positive) if positive else None
    reliability, contradiction_count, selective = _annotation_metrics(case, selected)
    coverage = None
    if threshold is not None and selected:
        rels = [case["rows"].get(item, {}).get("reliability", {x["candidate_id"]: x for x in case["menu"]}[item].get("reliability")) for item in selected]
        if all(value is not None for value in rels):
            coverage = sum(float(value) >= threshold for value in rels) / len(rels)
    oracle = _oracle(case)
    oracle_spent = None if oracle is None else _cost(case, oracle)
    oracle_loss = None if oracle is None else _loss(case, oracle)
    oracle_utility = None if oracle is None or oracle_loss is None or oracle_spent is None or rate is None else case["baseline_loss"] - oracle_loss - oracle_spent * rate
    regret = None if oracle_utility is None or utility is None else max(0.0, oracle_utility - utility)
    terminal = case.get("terminal_continuation_label")
    return {"schema_version": VERSION, "episode_id": case["episode_id"], "policy_id": policy_id, "seed": seed, "selected_candidates": list(selected), "selection_status": status, "stop_reason": status, "spent_cost": spent, "latency_ms": latency, "budget_remaining": None if case["budget"] is None or spent is None else max(0.0, case["budget"] - spent), "baseline_loss": case["baseline_loss"], "final_loss": final_loss, "decision_loss_reduction": reduction, "decision_loss_reduction_per_cost": ratio, "net_utility": utility, "regret_vs_oracle": regret, "final_selection_regret_proxy": regret, "oracle_utility": oracle_utility, "oracle_optimal": None if regret is None else regret <= _EPS, "candidate_precision": precision, "candidate_recall": recall, "positive_candidate_count": len(positive), "stop_quality": terminal if isinstance(terminal, bool) else None, "reliability_coverage": coverage, "reliability_mean": reliability, "contradiction_count": contradiction_count, "selective_risk_covered": selective, "evaluator_only": True, "runtime_safe": False, "value_baseline_type": "static_declared_value_baseline", "current_online_policy": False}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return math.fsum(values) / len(values) if values else None


def compare_policies(episodes: Iterable[Mapping[str, Any]], *, policies: Sequence[str] = POLICIES, seed: int = 0, reliability_threshold: float | None = None) -> dict[str, Any]:
    selected_policies = tuple(policies)
    if not selected_policies or len(set(selected_policies)) != len(selected_policies) or any(p not in POLICIES for p in selected_policies):
        raise ValueError("policies must be nonempty, unique, and known")
    raw_episodes = list(episodes)
    normalized = [normalize_episode(item) for item in raw_episodes]
    if not normalized:
        raise ValueError("at least one episode is required")
    ids = [case["episode_id"] for case in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode_id")
    rows_by_policy = {p: [replay_policy(raw_episodes[i], p, seed=seed + i, reliability_threshold=reliability_threshold) for i in range(len(raw_episodes))] for p in selected_policies}
    metrics = ("final_loss", "decision_loss_reduction", "decision_loss_reduction_per_cost", "net_utility", "regret_vs_oracle", "candidate_precision", "candidate_recall", "stop_quality", "reliability_coverage", "contradiction_count", "selective_risk_covered", "spent_cost", "latency_ms")
    aggregate: dict[str, dict[str, Any]] = {}
    for policy, rows in rows_by_policy.items():
        value: dict[str, Any] = {"policy_id": policy, "episode_count": len(rows)}
        for metric in metrics:
            observed = [row for row in rows if row.get(metric) is not None]
            value[f"mean_{metric}"] = _mean(observed, metric)
            value[f"{metric}_observed_denominator"] = len(observed)
            value[f"{metric}_observed_episode_ids"] = [row["episode_id"] for row in observed]
        value["decision_loss_reduction_per_cost"] = value["mean_decision_loss_reduction_per_cost"]
        value["mean_budget_remaining"] = _mean(rows, "budget_remaining")
        query_rows = [{"query_count": len(row["selected_candidates"])} for row in rows if isinstance(row.get("selected_candidates"), list)]
        value["mean_query_count"] = _mean(query_rows, "query_count")
        observed = [row for row in rows if row.get("regret_vs_oracle") is not None]
        value["policy_regret_rate"] = sum(row["regret_vs_oracle"] > _EPS for row in observed) / len(observed) if observed else None
        observed = [row for row in rows if row.get("stop_quality") is not None]
        value["stop_quality_rate"] = sum(bool(row["stop_quality"]) for row in observed) / len(observed) if observed else None
        observed = [row for row in rows if row.get("contradiction_count") is not None]
        value["contradiction_rate"] = sum(row["contradiction_count"] > 0 for row in observed) / len(observed) if observed else None
        observed = [row for row in rows if row.get("selective_risk_covered") is not None]
        value["selective_risk_coverage"] = _mean([{"v": float(row["selective_risk_covered"])} for row in observed], "v") if observed else None
        if "fixed_all" in rows_by_policy and policy != "fixed_all":
            fixed = {row["episode_id"]: row for row in rows_by_policy["fixed_all"]}
            paired = [(row, fixed[row["episode_id"]]) for row in rows if row["episode_id"] in fixed and row.get("spent_cost") is not None and fixed[row["episode_id"]].get("spent_cost") is not None]
            fixed_total = math.fsum(pair[1]["spent_cost"] for pair in paired)
            value["cost_savings_vs_fixed_all"] = None if fixed_total <= _EPS else 1.0 - math.fsum(pair[0]["spent_cost"] for pair in paired) / fixed_total
            value["cost_savings_vs_fixed_all_observed_denominator"] = len(paired)
        else:
            value["cost_savings_vs_fixed_all"] = None
            value["cost_savings_vs_fixed_all_observed_denominator"] = 0
        aggregate[policy] = value
    return {"schema_version": VERSION, "policy_order": list(selected_policies), "episode_ids": ids, "rows": rows_by_policy, "aggregate": aggregate, "contract": {"matched_episode_set": True, "paired_common_episode_set": True, "independent_evaluator_sidecar_required": True, "runtime_policy_reads_evaluator_labels": False, "numeric_voi_training_eligible": False, "synthetic_or_offline_only": True, "reliability_threshold": reliability_threshold}}


__all__ = ["VERSION", "POLICIES", "INDEPENDENT_GOLD", "normalize_episode", "replay_policy", "compare_policies"]
