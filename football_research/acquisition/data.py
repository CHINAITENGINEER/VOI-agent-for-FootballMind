"""Verified historical episodes and an isolated controlled-reveal environment.

The offline loader may hold labels and full features. A policy receives only
``visible_state`` and public action metadata, never the episode object itself.
This is controlled historical revelation, not a live network cost experiment.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_research.data_adapters.platform_history import verify_completed_export, load_completed_history_export
from football_research.data_adapters.batch_features import FeatureIndex, FeatureJoinError
from football_research.data_adapters.types import json_record
from .state import GROUP_FIELDS, parse_time, sanitize_state


def load_episodes(export_dir: str | Path, lead_hours: float = 24) -> tuple[list[dict], dict]:
    """Join exactly one prematch snapshot per completed regulation-time match."""
    if not math.isfinite(lead_hours) or lead_hours <= 0:
        raise ValueError("lead_hours must be finite and positive")
    verified = verify_completed_export(export_dir)
    if "features" not in verified.payloads:
        raise ValueError("Completed export has no feature payload")
    dataset = load_completed_history_export(export_dir)
    # Detect producer changes between the independently verified projections.
    again = verify_completed_export(export_dir)
    if again.manifest_sha256 != verified.manifest_sha256:
        raise ValueError("Export changed during episode assembly")
    snapshots = verified.rows("features")
    versions = {str(row["feature_version"]) for row in snapshots if row.get("feature_version")}
    if len(versions) != 1:
        raise ValueError("Episode export must contain exactly one feature version")
    feature_version = next(iter(versions))
    index = FeatureIndex.from_rows(snapshots, allowed_features=[f for fs in GROUP_FIELDS.values() for f in fs])
    episodes, rejected = [], []
    for match in sorted(dataset.matches, key=lambda m: (m.kickoff, m.match_id)):
        cutoff = match.kickoff - timedelta(hours=lead_hours)
        try:
            if match.information_mode not in {"historical_reconstruction", "operational"}:
                raise FeatureJoinError("unsupported_episode_information_mode")
            if match.outcome is None or match.result_available_at is None:
                raise FeatureJoinError("missing_finished_regulation_label")
            if match.fixture_available_at is None or match.fixture_available_at > cutoff:
                raise FeatureJoinError("fixture_unavailable_at_cutoff")
            joined = index.join(match, prediction_cutoff=cutoff, information_mode=match.information_mode,
                                feature_version=feature_version, allow_missing_values=True)
            row = json_record(match)
            row.update({field: joined.get(field) for fields in GROUP_FIELDS.values() for field in fields})
            row.update(prediction_cutoff=cutoff.isoformat(), feature_version=feature_version,
                       features_available_at=joined["features_available_at"].isoformat(), dataset_version=dataset.dataset_version)
            episodes.append(row)
        except FeatureJoinError as exc:
            rejected.append({"match_id": match.match_id, "prediction_cutoff": cutoff.isoformat(),
                             "reason": exc.reason, "detail": exc.detail})
    missing_groups = Counter()
    for row in episodes:
        missing_groups.update(name for name, fields in GROUP_FIELDS.items() if all(row[f] is None for f in fields))
    audit = {"schema_version": "controlled-episodes-1.0", "dataset_version": dataset.dataset_version,
             "platform_manifest_sha256": verified.manifest_sha256, "feature_version": feature_version,
             "lead_hours": lead_hours, "source_match_count": len(dataset.matches), "episode_count": len(episodes),
             "unique_matches": len({r["match_id"] for r in episodes}), "excluded_count": len(rejected),
             "exclusions": rejected, "all_null_groups": dict(missing_groups), "feature_index": index.report(),
             "dataset_audit": dataset.audit,
             "interpretation": "controlled_reveal_of_verified_historical_features_not_live_acquisition"}
    return episodes, audit


class ControlledEpisode:
    """Offline evaluator-owned environment; expose only returned state to policy.

    Python object privacy is an API boundary, not a security sandbox. Running
    arbitrary policy code with this object would violate the experiment design.
    """

    def __init__(self, row: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]):
        self.__row = deepcopy(dict(row))
        self.__actions = {}
        self.__performed: list[str] = []
        cutoff = parse_time(row["prediction_cutoff"])
        if parse_time(row["features_available_at"]) > cutoff:
            raise ValueError("Episode features are unavailable at prediction cutoff")
        for action in actions:
            ident = str(action["action_id"])
            groups = list(action["group_ids"])
            if not ident or ident in self.__actions or not groups or len(groups) != len(set(groups)) or any(g not in GROUP_FIELDS for g in groups):
                raise ValueError("Actions require unique IDs and known distinct group IDs")
            if not isinstance(action.get("available", True), bool):
                raise ValueError("Action available must be boolean")
            public = {"action_id": ident, "group_ids": groups, "available": action.get("available", True),
                      "cost": deepcopy(action.get("cost", {})), "cost_status": action.get("cost_status", "simulated")}
            for key in ("source", "tool", "capability", "version", "latency_ms", "request_id"):
                if key in action:
                    public[key] = deepcopy(action[key])
            self.__actions[ident] = public
        self.__state = {"match_id": str(row["match_id"]), "prediction_cutoff": cutoff.isoformat(),
                        "information_mode": row["information_mode"],
                        "base": {name: row[name] for name in ("competition_id", "home_team_id", "away_team_id", "kickoff")},
                        "groups": {name: {"status": "not_acquired", "values": {}} for name in GROUP_FIELDS}}
        self.__state = sanitize_state(self.__state)

    @property
    def performed_actions(self) -> tuple[str, ...]:
        return tuple(self.__performed)

    def visible_state(self) -> dict[str, Any]:
        return sanitize_state(self.__state)

    def legal_actions(self) -> list[dict[str, Any]]:
        return [deepcopy(action) for ident, action in self.__actions.items()
                if ident not in self.__performed and action["available"] and
                any(self.__state["groups"][g]["status"] == "not_acquired" for g in action["group_ids"])]

    def reveal(self, action_id: str) -> dict[str, Any]:
        if action_id not in {action["action_id"] for action in self.legal_actions()}:
            raise ValueError("Action unavailable, already executed, or reveals no new groups")
        action = self.__actions[action_id]
        proposed = deepcopy(self.__state)
        for group_id in action["group_ids"]:
            if self.__state["groups"][group_id]["status"] != "not_acquired":
                continue
            values = {field: self.__row.get(field) for field in GROUP_FIELDS[group_id]}
            if all(value is None for value in values.values()):
                group = {"status": "source_missing", "values": {}}
            else:
                digest_payload = {"match_id": self.__row["match_id"], "prediction_cutoff": self.__row["prediction_cutoff"],
                                  "group_id": group_id, "values": values, "version": self.__row["feature_version"]}
                digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False, default=str).encode()).hexdigest()
                group = {"status": "observed", "values": values,
                         "available_at": self.__row["features_available_at"],
                         "version": self.__row["feature_version"], "evidence_refs": ["sha256:" + digest]}
            proposed["groups"][group_id] = group
        # Validate the whole response before the action is marked completed.
        self.__state = sanitize_state(proposed)
        self.__performed.append(action_id)
        return self.visible_state()
