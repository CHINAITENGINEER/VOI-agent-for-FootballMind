"""Match-only input and initial state contract for autonomous FootballMind runs.

The user identifies a match.  The system owns fixture resolution, information
needs, source choice and stopping.  This module deliberately creates no
probabilities and makes no claim that an unknown is important before the
state-bound VoI policy evaluates it.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any, Mapping


VERSION = "autonomous-match-request-v1"
STATE_VERSION = "autonomous-match-state-v1"
POLICY_MODE = "conditional_semantic_voi_untrained_g"
CAPABILITY_BINDING_VERSION = "autonomous-capability-binding-v1"

# Human match input is intentionally permissive about the Latin separator.
# Team names remain opaque strings; only the separator is normalized here.
_MATCH_SEPARATOR = re.compile(r"\s+v(?:s)?\.?\s+", re.IGNORECASE)
_ALLOWED = frozenset({
    "home", "away", "competition", "kickoff", "prediction_cutoff",
    "match_id", "timezone", "mode",
})
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "home_score", "away_score",
    "final_score", "actual_result", "future_result", "winner", "label",
    "realized_loss", "decision_loss", "settlement", "post_match",
})

_INFORMATION_FAMILIES = (
    {
        "id": "qualification_constraint",
        "tool_family": "competition_state",
        "source_capabilities": ["standings", "competition_rules"],
        "question": "积分、晋级和赛制约束是否改变双方的最低必要结果？",
        "relevance": "可能改变双方的激励结构与风险偏好，但不能直接推出轮换或战术。",
        "prerequisites": [],
    },
    {
        "id": "rotation_lineup",
        "tool_family": "lineup_projection",
        "source_capabilities": ["official_lineup", "team_news"],
        "question": "预计首发、轮换和关键球员可用性是什么？",
        "relevance": "可能改变实际可用实力；必须与‘已晋级’等背景事实分开验证。",
        "prerequisites": ["qualification_constraint"],
    },
    {
        "id": "injury_availability",
        "tool_family": "availability",
        "source_capabilities": ["injury_report", "suspension_report", "official_lineup"],
        "question": "伤停、停赛和临场可用性中，哪些事实可能改变判断？",
        "relevance": "可能改变阵容与实力，但来源时效和球员重要性需要单独记录。",
        "prerequisites": [],
    },
    {
        "id": "tactical_intent",
        "tool_family": "tactical_context",
        "source_capabilities": ["press_conference", "team_news", "competition_state"],
        "question": "赛前可观察证据是否支持进攻、控风险或防守倾向？",
        "relevance": "可能改变比赛节奏与结果分布；不能由排名或晋级状态单独推断。",
        "prerequisites": ["qualification_constraint"],
    },
    {
        "id": "recent_form_strength",
        "tool_family": "team_strength",
        "source_capabilities": ["match_history", "xg_statistics", "elo"],
        "question": "近期表现、实力和高级统计是否仍是当前有效信息？",
        "relevance": "用于建立相对实力基线，并检查信息是否过时或跨赛事不可比。",
        "prerequisites": [],
    },
    {
        "id": "market_movement",
        "tool_family": "market_signal",
        "source_capabilities": ["odds_snapshot", "odds_history"],
        "question": "赔率或市场变化是否提供了与已有状态不同的时点信号？",
        "relevance": "可作为外部聚合信号，但不能自动等同于真实原因或结果概率。",
        "prerequisites": [],
    },
    {
        "id": "schedule_rest_weather",
        "tool_family": "situational_context",
        "source_capabilities": ["schedule", "travel", "weather"],
        "question": "赛程、休息、旅行或天气是否有足够影响力值得查询？",
        "relevance": "通常只有在极端或与状态交互时才有价值，应由 VoI 决定是否获取。",
        "prerequisites": [],
    },
)


def _text(value: Any, name: str, *, minimum: int = 1, maximum: int = 240) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = " ".join(value.split())
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} length is outside the allowed range")
    return value


def _time(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _scan_forbidden(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN:
                raise ValueError(f"{path}.{key} is outcome or evaluator data")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _split_match(text: str) -> tuple[str, str] | None:
    match = _MATCH_SEPARATOR.search(text)
    if match:
        home, away = text[:match.start()], text[match.end():]
        if home.strip() and away.strip():
            return home.strip(), away.strip()
    for separator in ("对阵", "对战", " 对 "):
        if separator in text:
            home, away = text.split(separator, 1)
            if home.strip() and away.strip():
                return home.strip(), away.strip()
    # Accept Chinese input without requiring a space, while avoiding arbitrary
    # prose being treated as a fixture.
    match = re.match(r"^(.{1,120}?)(?:对阵|对战)(.{1,120})$", text)
    return None if not match else (match.group(1).strip(), match.group(2).strip())


def normalize_match_request(value: str | Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Normalize a match-only request without asking the user for a research scope.

    A missing date is retained as an explicit resolution requirement.  The
    caller may resolve it from public sources; this function never fabricates a
    kickoff or treats a historical result as a pre-match fact.
    """
    if isinstance(value, str):
        raw: dict[str, Any] = {"match": value.strip()}
        pair = _split_match(raw["match"])
        if pair is None:
            raise ValueError("match must contain two teams separated by vs、对阵 or 对战")
        raw.update(home=pair[0], away=pair[1])
        raw.pop("match", None)
    elif isinstance(value, Mapping):
        _scan_forbidden(value)
        raw = dict(value)
        if "match" in raw:
            if not isinstance(raw["match"], str):
                raise ValueError("match must be a string")
            pair = _split_match(raw["match"])
            if pair is None:
                raise ValueError("match must contain two teams separated by vs、对阵 or 对战")
            raw.setdefault("home", pair[0])
            raw.setdefault("away", pair[1])
            raw.pop("match", None)
    else:
        raise ValueError("match request must be a string or mapping")

    unknown = set(raw) - _ALLOWED
    if unknown:
        raise ValueError("unsupported match fields: " + ", ".join(sorted(unknown)))
    home, away = _text(raw.get("home"), "home"), _text(raw.get("away"), "away")
    if home.casefold() == away.casefold():
        raise ValueError("home and away teams must differ")
    competition = None if raw.get("competition") is None else _text(raw["competition"], "competition")
    match_id = None if raw.get("match_id") is None else _text(raw["match_id"], "match_id", maximum=300)
    kickoff = None if raw.get("kickoff") is None else _time(raw["kickoff"], "kickoff")
    cutoff = None if raw.get("prediction_cutoff") is None else _time(raw["prediction_cutoff"], "prediction_cutoff")
    if kickoff and cutoff and cutoff >= kickoff:
        raise ValueError("prediction_cutoff must be earlier than kickoff")
    if raw.get("mode") not in (None, "prematch", "retrospective"):
        raise ValueError("mode must be prematch or retrospective")
    mode = raw.get("mode") or "prematch"
    if mode == "retrospective" and not kickoff:
        raise ValueError("retrospective mode requires kickoff")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)
    kickoff_dt = None if kickoff is None else datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    temporal_status = "unresolved" if kickoff is None else ("past" if kickoff_dt <= current else "upcoming")
    if mode == "prematch" and temporal_status == "past":
        raise ValueError("prematch request kickoff has already passed; use retrospective mode explicitly")
    normalized = {
        "schema": VERSION,
        "match": {"match_id": match_id, "home": home, "away": away, "competition": competition,
                   "kickoff": kickoff, "prediction_cutoff": cutoff},
        "mode": mode,
        "temporal": {"status": temporal_status,
                      "resolution_required": [] if kickoff else ["kickoff"],
                      "decision_boundary": cutoff or "unresolved"},
        "policy": {"mode": POLICY_MODE, "empirical_g_status": "deferred_timestamped_stream",
                    "numerical_voi_calibrated": False},
    }
    normalized["target_key"] = "match-" + sha256(json.dumps(
        normalized["match"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:24]
    return normalized


def _state_from_normalized(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """Create actor-visible known/unknown state from an already normalized request."""
    match = normalized["match"]
    known = [
        {"field": "home_team", "value": match["home"], "status": "verified_input", "source": "user_match_input"},
        {"field": "away_team", "value": match["away"], "status": "verified_input", "source": "user_match_input"},
    ]
    if match.get("competition"):
        known.append({"field": "competition", "value": match["competition"],
                      "status": "verified_input", "source": "user_match_input"})
    if match.get("kickoff"):
        known.append({"field": "kickoff", "value": match["kickoff"],
                      "status": "verified_input", "source": "user_match_input"})
    unknowns = []
    for item in _INFORMATION_FAMILIES:
        unknowns.append({"id": item["id"], "status": "unknown", "question": item["question"],
                         "decision_relevance": item["relevance"],
                         "prerequisites": list(item["prerequisites"]),
                         "tool_family": item["tool_family"],
                         "source_capabilities": list(item["source_capabilities"]),
                         "candidate_action_id": "query." + item["id"],
                         "attempts": 0, "evidence_ids": []})
    # This is a capability catalogue, not a priority list.  Values and costs
    # stay null until the current state and available sources are evaluated.
    candidate_catalog = [{
        "action_id": "query." + item["id"],
        "information_need": item["id"],
        "tool_family": item["tool_family"],
        "source_capabilities": list(item["source_capabilities"]),
        "prerequisites": list(item["prerequisites"]),
        "status": "unscored",
        "declared_cost": None,
        "conditional_value": None,
    } for item in _INFORMATION_FAMILIES]
    return {
        "schema": STATE_VERSION,
        "target_key": normalized["target_key"],
        "match": deepcopy(match),
        "policy": deepcopy(normalized["policy"]),
        "known_information": known,
        "unknown_information": unknowns,
        "candidate_catalog": candidate_catalog,
        "baseline": {"status": "not_computed", "probabilities": None,
                      "reason": "No outcome or model evidence is fabricated at input stage."},
        "next_step": {"owner": "agent", "action": "resolve_fixture_and_build_candidates",
                      "user_question": None},
        "contract": {
            "user_supplies": ["match_identity"],
            "agent_owns": ["fixture_resolution", "candidate_generation", "voi_selection", "stopping"],
            "no_user_scope_question": True,
        },
    }


def build_autonomous_initial_state(request: Mapping[str, Any] | str) -> dict[str, Any]:
    """Create actor-visible known/unknown state for a match-only run."""
    normalized = (deepcopy(dict(request)) if isinstance(request, Mapping)
                  and request.get("schema") == VERSION else normalize_match_request(request))
    return _state_from_normalized(normalized)


def build_match_bundle(value: str | Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Return immutable request/state material for a headless run bootstrap."""
    request = normalize_match_request(value, now=now)
    state = _state_from_normalized(request)
    body = {"schema": "autonomous-match-bundle-v1", "request": request, "state": state}
    body["bundle_digest"] = sha256(json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return body


def bind_source_capabilities(state: Mapping[str, Any], capabilities: Mapping[str, str]) -> dict[str, Any]:
    """Bind observed source capability health without assigning VoI or order.

    ``capabilities`` is a health snapshot supplied by the source layer.  A
    capability can be available, unknown, unavailable, stale or conflicting.
    The result only tells the Agent which routes are technically eligible; it
    never turns availability into value and never auto-resolves a prerequisite.
    """
    if not isinstance(state, Mapping) or state.get("schema") != STATE_VERSION:
        raise ValueError("state must be an autonomous-match-state-v1 object")
    if not isinstance(capabilities, Mapping):
        raise ValueError("capabilities must be a mapping")
    allowed = {"available", "unknown", "unavailable", "stale", "conflicting"}
    snapshot = {}
    for key, status in capabilities.items():
        if not isinstance(key, str) or not key.strip() or status not in allowed:
            raise ValueError("invalid source capability status")
        snapshot[key] = status
    result = deepcopy(dict(state))
    entries = []
    for raw in state.get("candidate_catalog", []):
        item = deepcopy(raw)
        sources = list(item.get("source_capabilities", []))
        source_states = {source: snapshot.get(source, "unknown") for source in sources}
        if any(status == "available" for status in source_states.values()):
            availability = "available"
        elif source_states and all(status == "unavailable" for status in source_states.values()):
            availability = "unavailable"
        elif any(status in {"stale", "conflicting"} for status in source_states.values()):
            availability = "stale_or_conflicting"
        else:
            availability = "unknown"
        item.update({"source_states": source_states, "availability": availability,
                     "eligibility": "unscored", "conditional_value": None})
        entries.append(item)
    result["candidate_catalog"] = entries
    result["source_capabilities"] = snapshot
    result["capability_binding"] = {
        "version": CAPABILITY_BINDING_VERSION,
        "value_scored": False,
        "priority_assigned": False,
        "prerequisites_auto_resolved": False,
    }
    return result
