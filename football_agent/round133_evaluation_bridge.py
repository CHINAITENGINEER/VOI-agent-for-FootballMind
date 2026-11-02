"""Round133 bridge from the Round132 action ledger to evaluation episodes.

Round132's adapter intentionally accepts an explicitly supplied public snapshot
without changing it.  The live Experience snapshot still contains two pieces
of runtime state that are useful to the actor but have evaluation-contract
meaning only after a selection: the legacy ``match_identity.label`` spelling
and the actor's prior ``question_return_assessments``.  This module keeps the
Round132 files frozen and applies the narrow public-boundary projection needed
by Round133.

The candidate menu is recovered from the operation's frozen ``proposal.json``.
Selection diagnostics are policy metadata and may be empty or incomplete; they
are not a substitute for the proposal menu.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .value_evaluation_contract import ledger_entry_to_evaluation_episode


VERSION = "round133-evaluation-bridge-v1"

# These fields are actor/runtime history, rather than the public state at the
# instant immediately before the current selection.  In particular, a prior
# return judgement contains an ``outcome`` and must not be carried into the
# current evaluator snapshot as if it were an independent target label.
_DROP_SNAPSHOT_FIELDS = frozenset({
    "question_return_assessments",
})

# Keep this copy local to the bridge so its sanitization remains stable even if
# the older contract module grows new forbidden names.  Keys are compared
# case-insensitively, matching the contract validator's recursive guard.
_DROP_RESULT_KEYS = frozenset({
    "outcome", "outcomes", "label", "labels", "target_label", "result",
    "score", "scores", "goals", "home_goals", "away_goals", "future_result",
    "hidden_return", "hidden_returns", "hidden_features", "full_snapshot",
    "post_state", "after_state", "response_body", "raw_data", "actual_result",
    "counterfactual_return", "decision_loss_after", "realized_gain",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def sanitize_public_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project a live Experience public snapshot onto the evaluator boundary.

    ``match_identity.label`` is a target identifier in this runtime (it names
    the fixture being investigated), not a result label.  Rename it to
    ``match_label`` so the evaluator's recursive result-field guard does not
    mistake it for an outcome.  Prior return assessments are removed as a
    unit: their nested ``outcome`` values describe an earlier acquisition and
    are not pre-action input for the current decision.

    The function copies JSON-compatible data and never mutates the runtime
    state or reads a provider/result channel.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("ROUND133_PUBLIC_SNAPSHOT_OBJECT_REQUIRED")

    def project(value: Any, *, path: tuple[str, ...] = ()) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for raw_key, child in value.items():
                key = str(raw_key)
                if key.casefold() in _DROP_SNAPSHOT_FIELDS:
                    continue
                if path == ("match_identity",) and key == "label":
                    result["match_label"] = project(child, path=path + (key,))
                    continue
                # A caller may supply nested legacy document/result fields.
                # Strip those keys recursively; the sole semantic exception is
                # the target identifier handled immediately above.
                if key.casefold() in _DROP_RESULT_KEYS:
                    continue
                result[key] = project(child, path=path + (key,))
            return result
        if isinstance(value, list):
            return [project(child, path=path + (str(i),))
                    for i, child in enumerate(value)]
        return deepcopy(value)

    projected = project(snapshot)
    # A JSON round-trip keeps the adapter deterministic and rejects accidental
    # NaN/non-JSON values before the digest is bound into the episode.
    return json.loads(json.dumps(projected, ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":"),
                                 allow_nan=False))


def _candidate_ids(proposal: Any) -> list[str]:
    """Extract the complete ordered candidate menu from one proposal record."""
    rows = proposal.get("candidates") if isinstance(proposal, Mapping) else proposal
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.strip() and candidate_id not in result:
            result.append(candidate_id)
    return result


def load_operation_proposals(
    folder: str | Path,
    operation_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Read frozen proposal records for ledger operations.

    This is a local archive reader only.  It does not inspect ``result.json``,
    call tools, or make network/model calls.  Missing proposal files are left
    out so callers can retain the Round132 adapter's partial menu for that row.
    """
    root = Path(folder)
    wanted = set(operation_ids) if operation_ids is not None else None
    found: dict[str, Any] = {}
    for path in sorted((root / "operations").glob("*/proposal.json")):
        operation_id = path.parent.name
        if wanted is not None and operation_id not in wanted:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"ROUND133_INVALID_PROPOSAL:{operation_id}") from exc
        found[operation_id] = value
    return found


def ledger_to_evaluation_episodes(
    ledger: Mapping[str, Any],
    *,
    proposals: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    target_key: str | None = None,
    prediction_cutoff: str | None = None,
) -> list[dict[str, Any]]:
    """Build Round133 episodes with a sanitized snapshot and frozen menu.

    ``proposals`` maps ``selection_operation_id`` to either a proposal object
    (with a ``candidates`` list) or directly to that list.  When a proposal is
    unavailable, the Round132 selected-ID fallback is retained and the episode
    remains explicitly incomplete.
    """
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("entries"), list):
        raise ValueError("ROUND133_LEDGER_ENTRIES_REQUIRED")
    proposals = proposals or {}
    episodes: list[dict[str, Any]] = []
    for entry in ledger["entries"]:
        if not isinstance(entry, Mapping):
            raise ValueError("ROUND133_LEDGER_ENTRY_OBJECT_REQUIRED")
        operation_id = entry.get("selection_operation_id")
        supplied_snapshot = entry.get("pre_action_snapshot")
        snapshot = (sanitize_public_snapshot(supplied_snapshot)
                    if isinstance(supplied_snapshot, Mapping) else None)
        episode = ledger_entry_to_evaluation_episode(
            entry,
            run_id=run_id or ledger.get("run_id"),
            target_key=target_key,
            prediction_cutoff=prediction_cutoff,
            public_snapshot=snapshot,
            ledger_digest=ledger.get("ledger_digest"),
            state_digest=ledger.get("state_digest"),
        )
        menu = _candidate_ids(proposals.get(operation_id))
        if menu:
            episode["pre_action"]["candidate_ids"] = menu
            episode["pre_action"]["candidate_generation_complete"] = True
            episode["pre_action"]["snapshot_digest"] = _digest(
                episode["pre_action"]["snapshot"])
            episode["ledger_binding"]["candidate_menu_digest"] = _digest(menu)
        episode["bridge_version"] = VERSION
        episodes.append(episode)
    return episodes


def build_evaluation_episodes_from_run(
    folder: str | Path,
    ledger: Mapping[str, Any],
    *,
    target_key: str | None = None,
    prediction_cutoff: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for a verified local run archive.

    ``ledger`` is supplied by the caller after Round132 replay verification;
    this function only reads the frozen proposal files needed to restore the
    finite menu and then delegates to :func:`ledger_to_evaluation_episodes`.
    """
    operation_ids = [
        entry.get("selection_operation_id")
        for entry in ledger.get("entries", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("selection_operation_id"), str)
    ] if isinstance(ledger, Mapping) else []
    proposals = load_operation_proposals(folder, operation_ids)
    return ledger_to_evaluation_episodes(
        ledger,
        proposals=proposals,
        target_key=target_key,
        prediction_cutoff=prediction_cutoff,
    )


__all__ = [
    "VERSION", "sanitize_public_snapshot", "load_operation_proposals",
    "ledger_to_evaluation_episodes", "build_evaluation_episodes_from_run",
]
