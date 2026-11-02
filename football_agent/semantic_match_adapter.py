"""Bind match-only input state to a pre-registered semantic scenario case."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from .autonomous_match_contract import build_match_bundle


VERSION = "semantic-match-adapter-v1"


def _name(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _time_key(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    return value.strip()


def _check_context_binding(case: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    """Reject an identity match that silently points at another fixture.

    Names alone are insufficient for a registered case: the same clubs can
    meet more than once, and a competition or cutoff can change the available
    information.  Missing request fields remain unresolved for the caller;
    supplied fields must agree with the case snapshot exactly after the
    timestamp normalization performed by ``build_match_bundle``.
    """
    case_match = case.get("match", {})
    if not isinstance(case_match, Mapping):
        return
    pairs = (
        ("match_id", "match_id"),
        ("competition", "competition"),
        ("kickoff", "kickoff"),
        ("prediction_cutoff", "prediction_cutoff"),
    )
    for request_key, case_key in pairs:
        request_value = requested.get(request_key)
        case_value = case_match.get(case_key)
        if request_value is None or case_value is None:
            continue
        left = _time_key(request_value) if request_key in {"kickoff", "prediction_cutoff"} else request_value
        right = _time_key(case_value) if case_key in {"kickoff", "prediction_cutoff"} else case_value
        if left != right:
            raise ValueError(f"registered semantic case {case_key} does not match requested fixture")


def attach_match_state(case: Mapping[str, Any], match: str | Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Attach an actor-visible autonomous state after checking the registered fixture.

    The case still owns its pre-registered sources and structured baseline.  A
    user match cannot silently switch that fixture or inject an outcome.
    """
    if not isinstance(case, Mapping):
        raise ValueError("case must be a mapping")
    bundle = build_match_bundle(match, now=now)
    profile = case.get("source_profile", {})
    teams = profile.get("teams", []) if isinstance(profile, Mapping) else []
    # Accept both the canonical provider name and the name a user sees in the
    # case export (for example ``Leeds`` as an alias of ``Leeds United``).
    # Keep the side attached to every alias so an alias cannot silently swap
    # home and away.  Colliding aliases are rejected rather than guessed.
    registered: dict[str, str] = {}
    for item in teams:
        if not isinstance(item, Mapping):
            continue
        side = item.get("side")
        aliases = [item.get("canonical_name"), item.get("input_name")]
        for alias in aliases:
            key = _name(alias)
            if not key:
                continue
            previous = registered.get(key)
            if previous is not None and previous != side:
                raise ValueError("registered semantic case contains colliding team aliases")
            registered[key] = side
    requested = bundle["request"]["match"]
    if _name(requested["home"]) not in registered or _name(requested["away"]) not in registered:
        raise ValueError("match input does not match the registered semantic case")
    if registered[_name(requested["home"])] != "home" or registered[_name(requested["away"])] != "away":
        raise ValueError("match input must preserve the registered semantic case home/away order")
    _check_context_binding(case, requested)
    result = deepcopy(dict(case))
    result["autonomous_adapter"] = {"version": VERSION, "target_key": bundle["request"]["target_key"],
                                    "policy_mode": bundle["request"]["policy"]["mode"]}
    result["autonomous_request"] = bundle["request"]
    result["autonomous_initial_state"] = bundle["state"]
    return result


__all__ = ["VERSION", "attach_match_state"]
