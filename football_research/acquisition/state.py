"""Small, value-isolating runtime projection for acquisition research.

This module accepts public evidence only. It never opens an evidence reference,
manifest, feature store or label file. Unknown fields do not reach predictors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Mapping

GROUP_FIELDS = {
    "elo": ("home_elo", "away_elo"),
    "recent_form": ("home_rolling_goals_for", "home_rolling_goals_against", "home_rolling_points",
                    "away_rolling_goals_for", "away_rolling_goals_against", "away_rolling_points"),
    "xg": ("home_rolling_xg_for", "home_rolling_xg_against", "away_rolling_xg_for", "away_rolling_xg_against"),
}
GROUP_IDS = tuple(GROUP_FIELDS)
STATUSES = ("observed", "not_acquired", "source_missing", "stale")


def parse_time(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamp requires explicit timezone")
    return result.astimezone(timezone.utc)


def mask_key(groups: Any) -> int:
    return sum(1 << i for i, name in enumerate(GROUP_IDS) if name in groups)


def sanitize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project an untrusted mapping to visible data; strip hidden values first.

    A non-observed group's payload is ignored in its entirety. Observed groups
    allow finite numeric fields only. Missing values remain explicit nulls; they
    are imputed only by the fitted subset model's training-window statistics.
    """
    cutoff = parse_time(state["prediction_cutoff"])
    base = state.get("base", {})
    kickoff = parse_time(base["kickoff"])
    if cutoff >= kickoff:
        raise ValueError("Acquisition prediction cutoff must precede kickoff")
    mode = str(state["information_mode"])
    if mode not in {"operational", "historical_reconstruction"}:
        raise ValueError("Unknown information mode")
    clean = {"match_id": str(state["match_id"]), "prediction_cutoff": cutoff.isoformat(),
             "information_mode": mode, "base": {name: str(base[name]) for name in
              ("competition_id", "home_team_id", "away_team_id")}, "groups": {}}
    if not clean["match_id"] or any(not v for v in clean["base"].values()):
        raise ValueError("Visible identifiers must be nonempty")
    clean["base"]["kickoff"] = kickoff.isoformat()
    visible = []
    for name, fields in GROUP_FIELDS.items():
        supplied = state.get("groups", {}).get(name, {"status": "not_acquired"})
        status = supplied.get("status", "not_acquired")
        if status not in STATUSES:
            raise ValueError("Unknown group status: " + str(status))
        group = {"status": status, "values": {}}
        if status == "observed":
            available = parse_time(supplied["available_at"])
            if available > cutoff:
                raise ValueError("Observed group is unavailable at prediction cutoff")
            values = supplied.get("values", {})
            for field in fields:
                value = values.get(field)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError("Observed fields must be finite numeric values or null")
                    value = float(value)
                group["values"][field] = value
            # Evidence summaries are inert digests/identifiers, never file paths.
            refs = supplied.get("evidence_refs", [])
            if not isinstance(refs, (list, tuple)) or any(not isinstance(v, str) or
                    not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", v) or re.match(r"^[A-Za-z]:", v) for v in refs):
                raise ValueError("Evidence references must be inert identifiers or content digests")
            group.update(available_at=available.isoformat(), version=str(supplied.get("version", "unspecified")),
                         evidence_refs=list(refs))
            visible.append(name)
        clean["groups"][name] = group
    clean["mask"] = visible
    clean["mask_key"] = mask_key(visible)
    return clean
