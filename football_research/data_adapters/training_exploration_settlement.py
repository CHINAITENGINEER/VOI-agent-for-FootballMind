"""Post-match settlement for the opt-in g training-exploration sidecar.

This module is intentionally a thin, auditable bridge around
``derive_g_candidates``.  It never re-computes timing, archive, or identity
gates, and it never enables the empirical g runtime.  Its job is to turn a
completed prospective trace into rows whose observed label is either a real
post-outcome gain or ``None``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from football_agent.g_label_contract import validate_transition

from .prospective_outcomes import derive_g_candidates


SETTLEMENT_SCHEMA = "footballmind-g-training-exploration-settlement-v1"
ADAPTER_SCHEMA = "footballmind-g-training-exploration-adapter-v1"


def _as_list(rows: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [row for row in (rows or ()) if isinstance(row, Mapping)]


def settle_training_exploration_rows(
    *,
    fixtures: Iterable[Mapping[str, Any]],
    discoveries: Iterable[Mapping[str, Any]],
    attempts: Iterable[Mapping[str, Any]],
    contexts: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    archive_root: str | Path | None = None,
    documents: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Settle a trace and return all actions plus an auditable report.

    ``derive_g_candidates`` remains the source of truth for the strict
    prospective gates and loss calculation.  We add explicit rows for actions
    that were visible but not selected so their label is visibly pending/null;
    such rows can never be mistaken for negative outcomes.
    """

    attempts_list = _as_list(attempts)
    derived = derive_g_candidates(
        fixtures=fixtures,
        discoveries=discoveries,
        attempts=attempts_list,
        contexts=contexts,
        results=results,
        archive_root=archive_root,
        documents=documents,
    )
    by_attempt = {row.get("attempt_id"): row for row in derived}
    rows: list[dict[str, Any]] = []
    counts = Counter()

    for attempt in attempts_list:
        attempt_id = attempt.get("attempt_id")
        selected = attempt.get("selected_by_policy") is True
        if selected:
            source = dict(by_attempt.get(attempt_id, {}))
            # A selected action should normally be represented by the strict
            # derivation.  Keep a defensive ineligible row if a malformed
            # attempt was filtered before derivation.
            if not source:
                source = {
                    "physical_match_id": attempt.get("physical_match_id"),
                    "attempt_id": attempt_id,
                    "action_id": attempt.get("action_id"),
                    "policy_arm": attempt.get("policy_arm"),
                    "selection_probability": attempt.get("selection_probability"),
                    "prediction_before": None,
                    "prediction_after": None,
                    "realized_gain": None,
                    "net_gain": None,
                    "label_status": "ineligible",
                    "exclusion_reason": ["selected_attempt_missing_from_derivation"],
                }
            gain = source.get("realized_gain")
            source["selected"] = True
            source["propensity"] = source.get("selection_probability")
            source["realized_signed_gain"] = gain
            source["label_status"] = (
                "prospective_realized" if source.get("label_status") == "eligible" and gain is not None
                else source.get("label_status", "ineligible")
            )
            if gain is not None:
                source["gain_direction"] = (
                    "positive_improvement" if gain > 0 else
                    "negative_harm" if gain < 0 else "zero_no_change"
                )
                counts["positive" if gain > 0 else "negative" if gain < 0 else "zero"] += 1
                counts["eligible"] += 1
            else:
                counts["selected_without_label"] += 1
            adapter = attempt.get("training_exploration")
            if isinstance(adapter, Mapping):
                source["training_exploration"] = {
                    "adapter_schema": adapter.get("adapter_schema", adapter.get("schema")),
                    "selection_authority": adapter.get("selection_authority"),
                    "label_status_at_collection": adapter.get("label_status", "pending_outcome"),
                }
            errors = validate_transition(source)
            if errors:
                source["label_validation_errors"] = errors
                counts["validation_errors"] += 1
            rows.append(source)
        else:
            # Include every visible/unselected action with an explicit null.
            # This is essential for later propensity-aware analysis: absence of
            # a query is not evidence that the query would have hurt.
            row = {
                "schema": SETTLEMENT_SCHEMA,
                "physical_match_id": attempt.get("physical_match_id"),
                "attempt_id": attempt_id,
                "action_id": attempt.get("action_id"),
                "selected": False,
                "selection_probability": attempt.get("selection_probability"),
                "propensity": attempt.get("selection_probability"),
                "prediction_before": None,
                "prediction_after": None,
                "realized_signed_gain": None,
                "label_status": "unexecuted",
                "training_eligible": False,
                "exclusion_reason": ["not_selected_or_not_executed"],
            }
            rows.append(row)
            counts["unexecuted_null"] += 1

    for row in rows:
        row.setdefault("schema", SETTLEMENT_SCHEMA)
        row.setdefault("training_eligible", row.get("label_status") == "prospective_realized")

    report = {
        "schema": SETTLEMENT_SCHEMA,
        "adapter_schema": ADAPTER_SCHEMA,
        "rows_seen": len(attempts_list),
        "rows_emitted": len(rows),
        "rows_eligible": counts["eligible"],
        "positive": counts["positive"],
        "negative": counts["negative"],
        "zero": counts["zero"],
        "selected_without_label": counts["selected_without_label"],
        "unexecuted_null": counts["unexecuted_null"],
        "validation_errors": counts["validation_errors"],
        "empirical_g_enabled": False,
        "selection_authority": "data_collection_only",
        "production_ready": False,
        "network_requests": 0,
        "label_definition": "loss_before_minus_loss_after_after_independent_post_match_outcome",
        "predicted_gain_is_not_label": True,
    }
    return rows, report


def write_settlement_package(
    output_dir: str | Path,
    *,
    rows: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Write a new settlement package without modifying the source run."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "g-training-rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    payload = dict(report)
    if source_manifest is not None:
        payload["source_manifest"] = dict(source_manifest)
    (out / "settlement-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# g training exploration settlement\n\n"
        "Observed labels are emitted only after strict prospective gates and an independent post-match result. "
        "Unselected actions retain `realized_signed_gain: null`; this package does not enable empirical g.\n",
        encoding="utf-8",
    )
    return out


__all__ = ["SETTLEMENT_SCHEMA", "settle_training_exploration_rows", "write_settlement_package"]
