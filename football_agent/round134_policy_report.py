"""Auditable reviewer report for the Round133 offline policy benchmark.

The benchmark replay is deliberately a low-level data structure.  This
module projects it into a reviewer-facing report without copying evaluator
sidecar rows or qualitative scenario expectations.  Every aggregate metric
keeps its observed denominator and every missing value remains explicit.

The report is an offline mechanism-evaluation artifact.  It describes which
information-selection policies queried what, how much resource they spent,
and where evaluation evidence is missing.  It does not claim improved
football results and it never changes Agent runtime state.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from .policy_benchmark import VERSION as POLICY_BENCHMARK_VERSION


VERSION = "round134-policy-report-v1"

# These are the decision/evaluation fields that the report intentionally
# tracks.  Keeping the list explicit prevents private sidecar fields from
# accidentally leaking into a reviewer projection.
TRACKED_METRICS = (
    "final_loss",
    "decision_loss_reduction",
    "net_utility",
    "regret_vs_oracle",
    "stop_quality",
    "reliability_coverage",
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_contract(benchmark: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = _mapping(benchmark.get("contract"), "benchmark.contract")
    if contract.get("synthetic_or_offline_only") is not True:
        raise ValueError("policy report requires a synthetic/offline benchmark")
    if contract.get("runtime_policy_reads_evaluator_labels") is not False:
        raise ValueError("runtime evaluator-label isolation is required")
    if contract.get("numeric_voi_training_eligible") is not False:
        raise ValueError("numeric VoI training eligibility must remain false")
    return contract


def _validate_benchmark(benchmark: Mapping[str, Any]) -> tuple[list[str], list[str], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark must be a mapping")
    if benchmark.get("schema_version") != POLICY_BENCHMARK_VERSION:
        raise ValueError("unexpected policy benchmark schema")
    contract = _validate_contract(benchmark)
    policy_order = _list(benchmark.get("policy_order"), "benchmark.policy_order")
    policy_order = [_text(value, "benchmark.policy_order[]") for value in policy_order]
    if not policy_order or len(set(policy_order)) != len(policy_order):
        raise ValueError("policy_order must be nonempty and unique")
    episode_ids = _list(benchmark.get("episode_ids"), "benchmark.episode_ids")
    episode_ids = [_text(value, "benchmark.episode_ids[]") for value in episode_ids]
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_ids must be nonempty and unique")
    rows_by_policy = _mapping(benchmark.get("rows"), "benchmark.rows")
    aggregate = _mapping(benchmark.get("aggregate"), "benchmark.aggregate")
    for policy in policy_order:
        rows = _list(rows_by_policy.get(policy), f"benchmark.rows[{policy}]")
        if len(rows) != len(episode_ids):
            raise ValueError(f"{policy} row count does not match episode_ids")
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"benchmark.rows[{policy}][{index}]")
            if row.get("episode_id") != episode_ids[index]:
                raise ValueError(f"{policy} episode ordering mismatch at index {index}")
            if row.get("policy_id") != policy:
                raise ValueError(f"{policy} policy ordering mismatch at index {index}")
        if policy not in aggregate:
            raise ValueError(f"aggregate missing policy {policy}")
        _mapping(aggregate[policy], f"benchmark.aggregate[{policy}]")
    return policy_order, episode_ids, rows_by_policy, aggregate, contract


def _observed(values: Sequence[Mapping[str, Any]], metric: str) -> list[Mapping[str, Any]]:
    return [row for row in values if row.get(metric) is not None]


def _coverage(values: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    denominator = len(values)
    observed = len(_observed(values, metric))
    unknown = denominator - observed
    return {
        "metric": metric,
        "observed_count": observed,
        "unknown_count": unknown,
        "denominator": denominator,
        "observed_rate": None if denominator == 0 else observed / denominator,
        "unknown_rate": None if denominator == 0 else unknown / denominator,
    }


def _project_episode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    selected = row.get("selected_candidates")
    if selected is None:
        selected_projection = None
        query_count = None
    elif isinstance(selected, list) and all(isinstance(value, str) for value in selected):
        selected_projection = list(selected)
        query_count = len(selected)
    else:
        raise ValueError("selected_candidates must be a list or null")
    stop_reason = row.get("stop_reason")
    if stop_reason is None:
        stop_reason = row.get("selection_status")
    if stop_reason is not None:
        stop_reason = _text(stop_reason, "stop_reason")
    unknown = [metric for metric in TRACKED_METRICS if row.get(metric) is None]
    return {
        "episode_id": _text(row.get("episode_id"), "episode_id"),
        "policy_id": _text(row.get("policy_id"), "policy_id"),
        "query_count": query_count,
        "selected_candidates": selected_projection,
        "resource_cost": row.get("spent_cost"),
        "resource_cost_observed": row.get("spent_cost") is not None,
        "stop_reason": stop_reason,
        "final_loss": row.get("final_loss"),
        "decision_loss_reduction": row.get("decision_loss_reduction"),
        "net_utility": row.get("net_utility"),
        "regret_vs_oracle": row.get("regret_vs_oracle"),
        "stop_quality": row.get("stop_quality"),
        "reliability_coverage": row.get("reliability_coverage"),
        "unknown_metrics": unknown,
        "unknown_metric_count": len(unknown),
        "unknown_metric_denominator": len(TRACKED_METRICS),
        "evaluator_only": row.get("evaluator_only") is True,
        "runtime_safe": row.get("runtime_safe") is True,
    }


def _aggregate_policy(
    policy: str,
    rows: Sequence[Mapping[str, Any]],
    source_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    projected = [_project_episode_row(row) for row in rows]
    denominator = len(projected)
    source = dict(source_aggregate)
    metric_coverage: dict[str, Any] = {}
    for metric in TRACKED_METRICS:
        coverage = _coverage(projected, metric)
        expected = source.get(f"{metric}_observed_denominator")
        if expected is not None:
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise ValueError(f"{policy}.{metric}_observed_denominator must be a nonnegative integer")
            if expected != coverage["observed_count"]:
                raise ValueError(f"{policy}.{metric}_observed_denominator is inconsistent")
        coverage["source_mean"] = source.get(f"mean_{metric}")
        coverage["source_observed_denominator"] = coverage["observed_count"]
        metric_coverage[metric] = coverage
    query_rows = [row for row in projected if row["query_count"] is not None]
    cost_rows = [row for row in projected if row["resource_cost"] is not None]
    stop_reasons = Counter(
        row["stop_reason"] if row["stop_reason"] is not None else "unknown_stop_reason"
        for row in projected
    )
    mean_query_count = None if not query_rows else sum(row["query_count"] for row in query_rows) / len(query_rows)
    mean_resource_cost = None if not cost_rows else sum(float(row["resource_cost"]) for row in cost_rows) / len(cost_rows)
    return {
        "policy_id": policy,
        "episode_count": denominator,
        "mean_query_count": mean_query_count,
        "query_count_observed_denominator": len(query_rows),
        "mean_resource_cost": mean_resource_cost,
        "resource_cost_observed_denominator": len(cost_rows),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "metric_coverage": metric_coverage,
        "regret": {
            "mean": source.get("mean_regret_vs_oracle"),
            "observed_denominator": metric_coverage["regret_vs_oracle"]["observed_count"],
            "unknown_count": metric_coverage["regret_vs_oracle"]["unknown_count"],
        },
        "stop_quality": {
            "mean": source.get("mean_stop_quality"),
            "observed_denominator": metric_coverage["stop_quality"]["observed_count"],
            "unknown_count": metric_coverage["stop_quality"]["unknown_count"],
        },
    }


def build_report(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Build a strict reviewer projection of ``compare_policies`` output.

    The returned object contains only policy/episode selections and explicit
    aggregate coverage.  Evaluator candidate rows, hidden outcomes, and
    scenario-author expectations are intentionally omitted.
    """
    policy_order, episode_ids, rows_by_policy, aggregate, contract = _validate_benchmark(benchmark)
    policies: dict[str, Any] = {}
    for policy in policy_order:
        rows = [_mapping(value, f"rows[{policy}][]") for value in rows_by_policy[policy]]
        projected_rows = [_project_episode_row(row) for row in rows]
        policies[policy] = {
            "summary": _aggregate_policy(policy, rows, _mapping(aggregate[policy], f"aggregate[{policy}]")),
            "episodes": projected_rows,
        }
    return {
        "schema_version": VERSION,
        "title": "FootballMind autonomous information-selection policy replay report",
        "source": {
            "schema_version": benchmark.get("schema_version"),
            "policy_order": list(policy_order),
            "episode_ids": list(episode_ids),
            "episode_count": len(episode_ids),
        },
        "policies": policies,
        "evidence_boundary": {
            "purpose": "offline_information_selection_mechanism_evaluation",
            "synthetic_or_offline_only": True,
            "world_truth_verified": False,
            "runtime_reads_evaluator_labels": False,
            "numeric_voi_training_eligible": False,
            "match_result_improvement_claim": False,
            "result_interpretation": "Offline policy replay evaluates information selection, resource use, stopping, and evidence coverage; it does not establish improved match results.",
        },
        "contract": {
            "policy_benchmark_schema": POLICY_BENCHMARK_VERSION,
            "matched_episode_set": contract.get("matched_episode_set") is True,
            "paired_common_episode_set": contract.get("paired_common_episode_set") is True,
            "independent_evaluator_sidecar_required": contract.get("independent_evaluator_sidecar_required") is True,
            "runtime_policy_reads_evaluator_labels": False,
            "synthetic_or_offline_only": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render an ASCII-stable reviewer report for logs and CI artifacts."""
    if not isinstance(report, Mapping) or report.get("schema_version") != VERSION:
        raise ValueError("unexpected policy report schema")
    boundary = _mapping(report.get("evidence_boundary"), "report.evidence_boundary")
    policies = _mapping(report.get("policies"), "report.policies")
    order = report.get("source", {}).get("policy_order", [])
    lines = [
        f"# {report.get('title', 'FootballMind policy replay report')}",
        "",
        "This report shows information-selection behavior: queried candidates, resource use, stopping reasons, and metrics that remain unknown.",
        "",
        "## Evidence boundary",
        "",
        f"- evaluation type: `{boundary.get('purpose')}`",
        "- evidence scope: offline/synthetic mechanism evaluation only.",
        "- match-result improvement claim: disabled.",
        "- numeric VoI training eligibility: disabled.",
        "- runtime evaluator-label access: disabled.",
        "",
        "## Policy summary",
        "",
        "| policy | episode count | mean query count (observed denominator) | mean resource cost (observed denominator) | stop reasons | mean regret (observed denominator) | mean stop quality (observed denominator) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for policy_id in order:
        item = _mapping(policies.get(policy_id), f"report.policies[{policy_id}]")
        summary = _mapping(item.get("summary"), f"report.policies[{policy_id}].summary")
        regret = _mapping(summary.get("regret"), "regret")
        stop = _mapping(summary.get("stop_quality"), "stop_quality")
        query = summary.get("mean_query_count")
        cost = summary.get("mean_resource_cost")
        query_display = f"{query} ({summary.get('query_count_observed_denominator')})" if query is not None else "unknown"
        cost_display = f"{cost} ({summary.get('resource_cost_observed_denominator')})" if cost is not None else "unknown"
        regret_display = f"{regret.get('mean')} ({regret.get('observed_denominator')})" if regret.get("mean") is not None else f"unknown (0/{summary.get('episode_count')})"
        stop_display = f"{stop.get('mean')} ({stop.get('observed_denominator')})" if stop.get("mean") is not None else f"unknown (0/{summary.get('episode_count')})"
        reasons = "; ".join(f"{key}={value}" for key, value in summary.get("stop_reason_counts", {}).items()) or "unknown"
        lines.append(f"| `{policy_id}` | {summary.get('episode_count')} | {query_display} | {cost_display} | {reasons} | {regret_display} | {stop_display} |")
    lines.extend(["", "## Episode replay", "", "| policy | episode | query count | resource cost | stop reason | regret | stop quality | unknown metrics |", "|---|---|---:|---:|---|---:|---:|---|"])
    for policy_id in order:
        item = _mapping(policies[policy_id], f"report.policies[{policy_id}]")
        for row in item.get("episodes", []):
            row = _mapping(row, "report episode")
            unknown = ", ".join(row.get("unknown_metrics", [])) or "-"
            lines.append(f"| `{policy_id}` | `{row.get('episode_id')}` | {row.get('query_count') if row.get('query_count') is not None else 'unknown'} | {row.get('resource_cost') if row.get('resource_cost') is not None else 'unknown'} | {row.get('stop_reason') or 'unknown'} | {row.get('regret_vs_oracle') if row.get('regret_vs_oracle') is not None else 'unknown'} | {row.get('stop_quality') if row.get('stop_quality') is not None else 'unknown'} | {unknown} |")
    lines.extend(["", "## Unknown coverage", "", "The denominator is the number of episodes with an observed value; unknown is never treated as zero.", ""])
    for policy_id in order:
        summary = _mapping(_mapping(policies[policy_id], "policy").get("summary"), "summary")
        lines.extend([f"### `{policy_id}`", "", "| metric | observed | unknown | denominator | observed rate |", "|---|---:|---:|---:|---:|"])
        for metric in TRACKED_METRICS:
            coverage = summary.get("metric_coverage", {}).get(metric, {})
            lines.append(f"| `{metric}` | {coverage.get('observed_count')} | {coverage.get('unknown_count')} | {coverage.get('denominator')} | {coverage.get('observed_rate')} |")
        lines.append("")
    lines.extend(["## Interpretation boundary", "", "This report audits autonomous query trajectories, cost, stopping, and evidence coverage. Synthetic evaluator values are not treated as real match labels and are not written to Agent runtime.", ""])
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    """Persist JSON and optional Markdown without changing runtime state."""
    if not isinstance(report, Mapping) or report.get("schema_version") != VERSION:
        raise ValueError("unexpected policy report schema")
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_target = Path(markdown_path)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.write_text(render_markdown(report), encoding="utf-8")


__all__ = ["VERSION", "TRACKED_METRICS", "build_report", "render_markdown", "write_report"]
