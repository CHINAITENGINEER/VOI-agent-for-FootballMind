"""Deterministic offline scenario fixtures for Round133-B.

The fixtures exercise FootballMind's *information-selection* problem rather
than a match-result model.  Every scenario has an as-of public snapshot, a
finite candidate menu and an evaluator-only policy expectation.  The actor
projection deliberately removes the latter so a policy can decide which
source to query and when to stop without being handed the answer.

This module is a data generator and validator only.  It performs no network
I/O, does not train a model, and does not turn qualitative expectations into a
numeric VoI label.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "footballmind-scenario-fixture-v1"
BUNDLE_VERSION = "footballmind-scenario-bundle-v1"

# These keys are rejected anywhere inside the actor-visible snapshot.  A
# result may exist in ``evaluation_reference`` for offline review, but never
# in the as-of state consumed by an autonomous policy.
FORBIDDEN_PUBLIC_KEYS = {
    "result", "results", "score", "scores", "goals", "home_goals",
    "away_goals", "outcome", "outcomes", "winner", "actual_result",
    "post_state", "after_state", "hidden_return", "decision_loss_after",
}
SOURCE_STATES = {"fresh", "stale", "missing", "duplicate", "conflict", "blocked"}
VALUE_CLASSES = {"high", "low", "conditional", "unusable"}
SCENARIO_AUTHOR_HYPOTHESIS = "SCENARIO_AUTHOR_HYPOTHESIS_NOT_GOLD"
PUBLIC_CANDIDATE_KEYS = {
    "candidate_id", "tool", "question", "source_ids", "source_state",
    "availability", "declared_cost", "requires",
}
ACTOR_FORBIDDEN_KEYS = FORBIDDEN_PUBLIC_KEYS | {
    "evaluation", "evaluation_reference", "expected_policy", "evaluation_value_class",
    "predicted_answer", "gold", "label_status", "decision_loss_before", "voi",
}
ACTOR_VIEW_KEYS = {
    "schema_version", "actor_view_version", "fixture_id", "target_key",
    "prediction_cutoff", "snapshot", "snapshot_digest", "candidate_menu",
}


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _opaque_fixture_id(label: str) -> str:
    """Keep fixture identity stable without publishing a semantic answer."""
    return "scenario-" + sha256(label.encode("utf-8")).hexdigest()[:20]


def _unique_strings(value: Any, *, nonempty: bool = True) -> bool:
    if not isinstance(value, list):
        return False
    if any(not isinstance(item, str) or (nonempty and not item.strip()) for item in value):
        return False
    return len(value) == len(set(value))


def _iso(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("scenario fixture times must be timezone-aware")
    return value


def _candidate(candidate_id: str, tool: str, question: str, *, source_ids: list[str],
               source_state: str = "fresh", cost: int = 1,
               value_class: str = "conditional", availability: str = "available",
               requires: list[str] | None = None) -> dict[str, Any]:
    if source_state not in SOURCE_STATES:
        raise ValueError(f"unknown source state: {source_state}")
    if value_class not in VALUE_CLASSES:
        raise ValueError(f"unknown value class: {value_class}")
    return {
        "candidate_id": candidate_id,
        "tool": tool,
        "question": question,
        "source_ids": list(source_ids),
        "source_state": source_state,
        "availability": availability,
        "declared_cost": {"external_calls": cost},
        "requires": list(requires or []),
        # Evaluator-only fields are stripped by actor_projection().
        "evaluation_value_class": value_class,
    }


def _base(fixture_id: str, title: str, family: str, *, cutoff: str,
          kickoff: str, match_id: str, home: str, away: str,
          known: Mapping[str, Any], unknown: list[str],
          sources: list[Mapping[str, Any]], candidates: list[Mapping[str, Any]],
          high: list[str], low: list[str], first: list[str], stop_after: list[str],
          stop_condition: str, rationale: str,
          reference: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = {
        "match": {
            "match_id": match_id,
            "competition": known.get("competition", "synthetic-competition"),
            "home_team": home,
            "away_team": away,
            "kickoff_at": kickoff,
        },
        "known_information": dict(known),
        "open_questions": list(unknown),
        "source_observations": [dict(item) for item in sources],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": _opaque_fixture_id(fixture_id),
        "title": title,
        "scenario_family": family,
        "synthetic": True,
        "target_key": f"match:{match_id}",
        "as_of": {"prediction_cutoff": _iso(cutoff), "timezone": "UTC"},
        "public_snapshot": snapshot,
        "candidate_menu": [dict(item) for item in candidates],
        "expected_policy": {
            "high_value_candidates": list(high),
            "low_value_candidates": list(low),
            "expected_first_selection": list(first),
            "stop_after_candidates": list(stop_after),
            "stop_condition": stop_condition,
            "rationale": rationale,
            "label_status": SCENARIO_AUTHOR_HYPOTHESIS,
            "gold": False,
        },
        # This section is sidecar material for an offline evaluator.  It is
        # deliberately hidden by actor_projection().
        "evaluation_reference": dict(reference),
    }


def _scenario_fixtures() -> list[dict[str, Any]]:
    """Build the fixed Round133-B scenario set in stable order."""
    # All timestamps are synthetic, as-of, timezone-aware values.  They are
    # not claims about a live fixture or a training observation.
    return [
        _base(
            "qualified-but-strong-performance", "已出线信息不应强行改变判断",
            "qualification_signal_low_value", cutoff="2026-09-30T10:00:00Z",
            kickoff="2026-09-30T12:00:00Z", match_id="syn-jpn-tha-001",
            home="Japan", away="Thailand",
            known={"competition": "synthetic-asian-games-group", "home_already_qualified": True,
                   "table_gap_is_material": True, "baseline_strength_signal": "home_favored"},
            unknown=["是否轮换主力", "最终首发是否改变实力差", "天气是否会影响节奏"],
            sources=[
                {"source_id": "standings", "kind": "competition_state", "status": "fresh",
                 "observed_at": "2026-09-30T09:40:00Z"},
                {"source_id": "team-news", "kind": "lineup_news", "status": "fresh",
                 "observed_at": "2026-09-30T09:50:00Z"},
                {"source_id": "weather", "kind": "weather", "status": "fresh",
                 "observed_at": "2026-09-30T09:30:00Z"},
            ],
            candidates=[
                _candidate("rotation_check", "lineup_news", "确认已出线后是否实际轮换主力",
                           source_ids=["team-news"], value_class="conditional"),
                _candidate("weather_check", "weather_lookup", "确认天气是否足以改变比赛节奏",
                           source_ids=["weather"], value_class="low"),
                _candidate("standings_refresh", "standings_lookup", "复核出线状态与淘汰赛规则",
                           source_ids=["standings"], value_class="low"),
            ],
            high=["rotation_check"], low=["weather_check", "standings_refresh"],
            first=["rotation_check"], stop_after=["rotation_check"],
            stop_condition="若首发/轮换没有产生实质实力变化，保留基础判断并停止；不要因已出线标签自动下调主队。",
            rationale="资格状态是一个可能有用的候选线索，但不是轮换或弱化的充分条件；信息价值取决于是否发现实际阵容变化。",
            reference={"actor_visible": False,
                       "provenance": "user_supplied_unverified",
                       "evidence_role": "motivation_only",
                       "not_gold": True,
                       "observed_pattern": "Japan later won 3-0 in the motivating example",
                       "expected_behavior": "qualification alone does not force a negative update",
                       "claim_scope": "A single match result does not prove that qualification information has low value before kickoff."},
        ),
        _base(
            "draw-secures-group-first", "平局即可第一时竞争目标成为高价值信息",
            "competition_constraint_high_value", cutoff="2026-09-30T11:00:00Z",
            kickoff="2026-09-30T13:00:00Z", match_id="syn-chn-uae-001",
            home="China", away="United Arab Emirates",
            known={"competition": "synthetic-asian-games-group", "points_gap": 1,
                   "current_rank_uncertain": True, "baseline_strength_signal": "balanced"},
            unknown=["平局是否确保小组第一", "同分排名规则是什么", "球队会采用何种比赛策略"],
            sources=[
                {"source_id": "table-live", "kind": "standings", "status": "fresh",
                 "observed_at": "2026-09-30T10:30:00Z"},
                {"source_id": "competition-rules", "kind": "regulations", "status": "fresh",
                 "observed_at": "2026-09-30T10:20:00Z"},
                {"source_id": "team-news", "kind": "tactical-news", "status": "fresh",
                 "observed_at": "2026-09-30T10:45:00Z"},
                {"source_id": "weather", "kind": "weather", "status": "fresh",
                 "observed_at": "2026-09-30T10:30:00Z"},
            ],
            candidates=[
                _candidate("qualification_constraint", "standings_and_rules", "确认平局、积分和同分规则能否锁定小组第一",
                           source_ids=["table-live", "competition-rules"], value_class="high"),
                _candidate("tactical_intent", "team_news", "确认球队是否以守平为主要策略",
                           source_ids=["team-news"], value_class="high", requires=["qualification_constraint"]),
                _candidate("weather_check", "weather_lookup", "确认天气是否改变攻守策略",
                           source_ids=["weather"], value_class="low"),
            ],
            high=["qualification_constraint", "tactical_intent"], low=["weather_check"],
            first=["qualification_constraint"], stop_after=["tactical_intent"],
            stop_condition="确认竞争约束和战术意图后，若新增普通状态信息不再改变决策，停止继续查询。",
            rationale="排名/晋级规则直接改变决策目标；它比泛化的近期状态更可能降低当前风险，应优先查询。",
            reference={"actor_visible": False, "expected_behavior": "constraint-first, then tactical intent"},
        ),
        _base(
            "key-forward-injury", "关键前锋伤停是结果敏感信息",
            "injury_high_value", cutoff="2026-09-30T12:00:00Z",
            kickoff="2026-09-30T15:00:00Z", match_id="syn-injury-001",
            home="Alpha FC", away="Beta FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "home_favored",
                   "key_forward_status": "unknown", "recent_attack_signal": "unstable"},
            unknown=["关键前锋能否出场", "替补能力是否足够覆盖缺口", "天气是否影响进攻"],
            sources=[
                {"source_id": "club-medical", "kind": "injury_report", "status": "fresh",
                 "observed_at": "2026-09-30T11:50:00Z"},
                {"source_id": "lineup-official", "kind": "lineup", "status": "fresh",
                 "observed_at": "2026-09-30T11:55:00Z"},
                {"source_id": "weather", "kind": "weather", "status": "fresh",
                 "observed_at": "2026-09-30T11:00:00Z"},
            ],
            candidates=[
                _candidate("key_forward_availability", "injury_lookup", "核实关键前锋伤停和预计出场状态",
                           source_ids=["club-medical", "lineup-official"], value_class="high"),
                _candidate("replacement_quality", "squad_depth_lookup", "评估替补对进攻强度的覆盖程度",
                           source_ids=["lineup-official"], value_class="conditional", requires=["key_forward_availability"]),
                _candidate("weather_check", "weather_lookup", "确认天气是否会放大进攻缺口",
                           source_ids=["weather"], value_class="low"),
            ],
            high=["key_forward_availability"], low=["weather_check"],
            first=["key_forward_availability"], stop_after=["replacement_quality"],
            stop_condition="确认前锋状态及必要的替补覆盖后停止；天气只在极端预报时重新开放。",
            rationale="伤停状态与当前攻击不稳定直接相连，是高影响、低歧义的下一查询。",
            reference={"actor_visible": False, "expected_behavior": "injury-first"},
        ),
        _base(
            "weather-low-value", "天气正常时的低价值候选",
            "weather_low_value", cutoff="2026-09-30T13:00:00Z",
            kickoff="2026-09-30T16:00:00Z", match_id="syn-weather-001",
            home="Gamma FC", away="Delta FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "away_favored",
                   "weather_preview": "mild_and_dry", "forecast_confidence": "high"},
            unknown=["是否有临场极端天气", "阵容是否存在未确认变化"],
            sources=[
                {"source_id": "forecast", "kind": "weather", "status": "fresh",
                 "observed_at": "2026-09-30T12:45:00Z"},
                {"source_id": "team-news", "kind": "lineup_news", "status": "fresh",
                 "observed_at": "2026-09-30T12:50:00Z"},
            ],
            candidates=[
                _candidate("weather_detail", "weather_lookup", "查询小时级天气与场地影响",
                           source_ids=["forecast"], value_class="low"),
                _candidate("lineup_check", "lineup_news", "确认是否存在会改变强弱的阵容变化",
                           source_ids=["team-news"], value_class="conditional"),
            ],
            high=[], low=["weather_detail"], first=["lineup_check"], stop_after=["lineup_check"],
            stop_condition="正常天气且阵容无重大变化时立即停止；不为填满工具调用而查询天气细节。",
            rationale="已知天气温和且置信度高，边际信息价值低于一次外部调用成本。",
            reference={"actor_visible": False, "expected_behavior": "skip weather and stop after relevant lineup check"},
        ),
        _base(
            "odds-news-conflict", "赔率与新闻冲突时先做独立核验",
            "market_news_conflict", cutoff="2026-09-30T14:00:00Z",
            kickoff="2026-09-30T17:00:00Z", match_id="syn-conflict-001",
            home="Epsilon FC", away="Zeta FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "balanced",
                   "market_move": "home_shortening", "news_signal": "home_rotation_rumor"},
            unknown=["赔率变动是否由可靠信息驱动", "轮换传闻是否有独立证据", "是否存在市场噪声"],
            sources=[
                {"source_id": "odds-a", "kind": "market_odds", "status": "fresh",
                 "observed_at": "2026-09-30T13:55:00Z"},
                {"source_id": "news-a", "kind": "team_news", "status": "conflict",
                 "observed_at": "2026-09-30T13:40:00Z"},
                {"source_id": "lineup-official", "kind": "lineup", "status": "fresh",
                 "observed_at": "2026-09-30T13:50:00Z"},
                {"source_id": "weather", "kind": "weather", "status": "fresh",
                 "observed_at": "2026-09-30T13:30:00Z"},
            ],
            candidates=[
                _candidate("independent_lineup_verify", "lineup_news", "寻找官方或独立来源确认轮换传闻",
                           source_ids=["lineup-official"], value_class="high"),
                _candidate("odds_movement_context", "odds_history", "查询赔率变动时间线与多庄一致性",
                           source_ids=["odds-a"], value_class="conditional"),
                _candidate("weather_check", "weather_lookup", "确认天气是否解释市场变动",
                           source_ids=["weather"], value_class="low"),
            ],
            high=["independent_lineup_verify"], low=["weather_check"],
            first=["independent_lineup_verify"], stop_after=["odds_movement_context"],
            stop_condition="冲突被独立阵容证据解释且赔率时间线不再产生新分支时停止；未核验传闻不得直接改写预测。",
            rationale="冲突提高了错误更新风险，独立核验的预期风险降低高于额外天气查询。",
            reference={"actor_visible": False, "expected_behavior": "verify conflict before updating"},
        ),
        _base(
            "stale-source", "过期来源应被识别并寻找新鲜替代",
            "source_staleness", cutoff="2026-09-30T15:00:00Z",
            kickoff="2026-09-30T18:00:00Z", match_id="syn-stale-001",
            home="Eta FC", away="Theta FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "balanced",
                   "last_lineup_update_age_minutes": 240},
            unknown=["阵容是否在最近一小时发生变化", "旧来源是否仍然有效"],
            sources=[
                {"source_id": "old-lineup", "kind": "lineup", "status": "stale",
                 "observed_at": "2026-09-30T11:00:00Z"},
                {"source_id": "official-feed", "kind": "lineup", "status": "fresh",
                 "observed_at": "2026-09-30T14:55:00Z"},
            ],
            candidates=[
                _candidate("fresh_lineup", "official_lineup", "读取临近开赛的官方阵容状态",
                           source_ids=["official-feed"], source_state="fresh", value_class="high"),
                _candidate("reuse_old_lineup", "cached_lineup", "重复使用旧阵容快照",
                           source_ids=["old-lineup"], source_state="stale", value_class="unusable"),
            ],
            high=["fresh_lineup"], low=["reuse_old_lineup"], first=["fresh_lineup"],
            stop_after=["fresh_lineup"],
            stop_condition="旧来源不得满足新鲜度约束；读取新鲜阵容后若无重大变化即可停止。",
            rationale="过期不是低价值，而是不可直接使用的质量状态；Agent 应付出一次替代查询成本。",
            reference={"actor_visible": False, "expected_behavior": "reject stale, query fresh"},
        ),
        _base(
            "missing-source", "来源缺失时不把空响应当作事实",
            "source_missing", cutoff="2026-09-30T16:00:00Z",
            kickoff="2026-09-30T19:00:00Z", match_id="syn-missing-001",
            home="Iota FC", away="Kappa FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "home_favored",
                   "injury_feed_status": "unavailable"},
            unknown=["关键球员是否缺阵", "是否存在替代伤停来源"],
            sources=[
                {"source_id": "injury-feed", "kind": "injury", "status": "missing",
                 "observed_at": "2026-09-30T15:40:00Z"},
                {"source_id": "club-news", "kind": "team_news", "status": "fresh",
                 "observed_at": "2026-09-30T15:50:00Z"},
            ],
            candidates=[
                _candidate("club_news_injury", "club_news", "查找俱乐部公告中的伤停信息",
                           source_ids=["club-news"], value_class="high"),
                _candidate("missing_feed_retry", "injury_feed", "重试当前不可用的伤停源",
                           source_ids=["injury-feed"], source_state="missing", availability="unavailable",
                           value_class="unusable"),
            ],
            high=["club_news_injury"], low=["missing_feed_retry"], first=["club_news_injury"],
            stop_after=["club_news_injury"],
            stop_condition="若替代来源也不可用，显式保留未知并停止；空响应不构造‘无伤停’结论。",
            rationale="缺失是证据状态而非负面事实；一次可用替代来源查询比重复失败更有价值。",
            reference={"actor_visible": False, "expected_behavior": "use alternate source, preserve unknown on failure"},
        ),
        _base(
            "duplicate-source", "重复来源不应伪造独立确认",
            "source_duplicate", cutoff="2026-09-30T17:00:00Z",
            kickoff="2026-09-30T20:00:00Z", match_id="syn-duplicate-001",
            home="Lambda FC", away="Mu FC",
            known={"competition": "synthetic-league", "baseline_strength_signal": "balanced",
                   "reported_injury": "one_unverified_report"},
            unknown=["伤停报道是否有独立来源", "两份报道是否来自同一上游"],
            sources=[
                {"source_id": "aggregator-a", "kind": "injury", "status": "duplicate",
                 "observed_at": "2026-09-30T16:30:00Z", "content_digest": "d" * 64},
                {"source_id": "aggregator-b", "kind": "injury", "status": "duplicate",
                 "observed_at": "2026-09-30T16:35:00Z", "content_digest": "d" * 64},
                {"source_id": "club-official", "kind": "club_news", "status": "fresh",
                 "observed_at": "2026-09-30T16:55:00Z"},
            ],
            candidates=[
                _candidate("independent_injury_confirm", "club_news", "寻找不共享上游的俱乐部确认",
                           source_ids=["club-official"], value_class="high"),
                _candidate("count_duplicate_reports", "source_dedup", "把重复报道数量当作置信度",
                           source_ids=["aggregator-a", "aggregator-b"], source_state="duplicate",
                           value_class="unusable"),
            ],
            high=["independent_injury_confirm"], low=["count_duplicate_reports"],
            first=["independent_injury_confirm"], stop_after=["independent_injury_confirm"],
            stop_condition="同一内容摘要的多个来源只算一个证据；获得独立确认或明确保留未知后停止。",
            rationale="重复数量不等于独立支持，独立来源查询才有降低不确定性的机会。",
            reference={"actor_visible": False, "expected_behavior": "deduplicate before valuing evidence"},
        ),
    ]


def validate_scenario_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic validation report for one scenario fixture."""
    reasons: list[str] = []
    if not isinstance(fixture, Mapping):
        return {"valid": False, "reasons": ["fixture must be an object"]}
    required = {"schema_version", "fixture_id", "target_key", "as_of", "public_snapshot",
                "candidate_menu", "expected_policy", "evaluation_reference"}
    reasons.extend(f"missing:{key}" for key in sorted(required - set(fixture)))
    if fixture.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if not isinstance(fixture.get("fixture_id"), str) or not fixture.get("fixture_id"):
        reasons.append("fixture_id_required")
    as_of = fixture.get("as_of")
    cutoff = None
    if not isinstance(as_of, Mapping):
        reasons.append("as_of_object_required")
    else:
        try:
            cutoff = datetime.fromisoformat(as_of.get("prediction_cutoff", "").replace("Z", "+00:00"))
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError
            cutoff = cutoff.astimezone(timezone.utc)
        except (TypeError, ValueError):
            reasons.append("invalid_prediction_cutoff")
    snapshot = fixture.get("public_snapshot")
    if not isinstance(snapshot, Mapping):
        reasons.append("public_snapshot_object_required")
    else:
        leaked: list[str] = []
        def walk(value: Any, path: str = "public_snapshot") -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    key_text = str(key)
                    if key_text.casefold() in FORBIDDEN_PUBLIC_KEYS:
                        leaked.append(f"{path}/{key_text}")
                    walk(child, f"{path}/{key_text}")
            elif isinstance(value, list):
                for idx, child in enumerate(value):
                    walk(child, f"{path}/{idx}")
        walk(snapshot)
        reasons.extend(f"public_snapshot_result_leak:{path}" for path in leaked)
    sources = snapshot.get("source_observations", []) if isinstance(snapshot, Mapping) else []
    source_ids: set[str] = set()
    if not isinstance(sources, list):
        reasons.append("source_observations_list_required")
        sources = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str) or not source.get("source_id"):
            reasons.append(f"source_{index}_id_required")
            continue
        source_id = source["source_id"]
        if source_id in source_ids:
            reasons.append(f"duplicate_source:{source_id}")
        source_ids.add(source_id)
        try:
            observed = datetime.fromisoformat(source.get("observed_at", "").replace("Z", "+00:00"))
            if observed.tzinfo is None or observed.utcoffset() is None:
                raise ValueError
            if cutoff is not None and observed.astimezone(timezone.utc) > cutoff:
                reasons.append(f"source_after_cutoff:{source_id}")
        except (TypeError, ValueError):
            reasons.append(f"source_{source_id}_observed_at_invalid")

    candidates = fixture.get("candidate_menu")
    candidate_ids: list[str] = []
    if not isinstance(candidates, list) or not candidates:
        reasons.append("candidate_menu_required")
        candidates = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            reasons.append(f"candidate_{index}_object_required")
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            reasons.append(f"candidate_{index}_id_required")
        elif candidate_id in candidate_ids:
            reasons.append(f"duplicate_candidate:{candidate_id}")
        else:
            candidate_ids.append(candidate_id)
        extra_private = set(candidate) - PUBLIC_CANDIDATE_KEYS - {"evaluation_value_class"}
        if extra_private:
            reasons.append(f"candidate_{candidate_id}_private_fields:{','.join(sorted(extra_private))}")
        state = candidate.get("source_state")
        if state not in SOURCE_STATES:
            reasons.append(f"candidate_{candidate_id}_source_state_invalid")
        value_class = candidate.get("evaluation_value_class")
        if value_class not in VALUE_CLASSES:
            reasons.append(f"candidate_{candidate_id}_value_class_invalid")
        cost = candidate.get("declared_cost")
        if (not isinstance(cost, Mapping) or not cost or
                any(not isinstance(key, str) or not key.strip() or type(v) is not int or v < 0
                    for key, v in cost.items())):
            reasons.append(f"candidate_{candidate_id}_cost_invalid")
        candidate_sources = candidate.get("source_ids")
        if not _unique_strings(candidate_sources):
            reasons.append(f"candidate_{candidate_id}_source_ids_invalid")
        elif any(item not in source_ids for item in candidate_sources):
            reasons.append(f"candidate_{candidate_id}_source_reference_missing")
        requires = candidate.get("requires", [])
        if not _unique_strings(requires, nonempty=False):
            reasons.append(f"candidate_{candidate_id}_requires_invalid")
        elif candidate_id in requires:
            reasons.append(f"candidate_{candidate_id}_requires_self")
    expected = fixture.get("expected_policy")
    if not isinstance(expected, Mapping):
        reasons.append("expected_policy_object_required")
        expected = {}
    for key in ("high_value_candidates", "low_value_candidates", "expected_first_selection", "stop_after_candidates"):
        values = expected.get(key)
        if not isinstance(values, list) or any(value not in candidate_ids for value in values):
            reasons.append(f"expected_{key}_outside_menu")
    # Dependencies must point to existing candidates and form a DAG.  This
    # keeps a policy replay from accepting an impossible acquisition plan.
    candidate_map = {
        item.get("candidate_id"): item for item in candidates
        if isinstance(item, Mapping) and isinstance(item.get("candidate_id"), str)
    }
    for candidate_id, candidate in candidate_map.items():
        for required_id in candidate.get("requires", []) if isinstance(candidate.get("requires", []), list) else []:
            if not isinstance(required_id, str) or required_id not in candidate_map:
                reasons.append(f"candidate_{candidate_id}_requires_missing:{required_id}")
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(candidate_id: str) -> None:
        if candidate_id in visiting:
            reasons.append(f"candidate_dependency_cycle:{candidate_id}")
            return
        if candidate_id in visited:
            return
        visiting.add(candidate_id)
        candidate = candidate_map.get(candidate_id, {})
        for required_id in candidate.get("requires", []) if isinstance(candidate.get("requires", []), list) else []:
            if isinstance(required_id, str) and required_id in candidate_map:
                visit(required_id)
        visiting.remove(candidate_id)
        visited.add(candidate_id)
    for candidate_id in candidate_map:
        visit(candidate_id)
    reference = fixture.get("evaluation_reference")
    if not isinstance(reference, Mapping) or reference.get("actor_visible") is not False:
        reasons.append("evaluation_reference_must_be_hidden")
    expected = fixture.get("expected_policy")
    if isinstance(expected, Mapping) and (expected.get("label_status") != SCENARIO_AUTHOR_HYPOTHESIS or expected.get("gold") is not False):
        reasons.append("expected_policy_must_be_non_gold_hypothesis")
    if isinstance(reference, Mapping) and reference.get("provenance") == "user_supplied_unverified":
        if reference.get("not_gold") is not True or reference.get("evidence_role") != "motivation_only":
            reasons.append("user_motivation_must_be_unverified_non_gold")
    return {
        "valid": not reasons,
        "fixture_id": fixture.get("fixture_id"),
        "candidate_count": len(candidate_ids),
        "snapshot_digest": _digest(snapshot) if isinstance(snapshot, Mapping) else None,
        "reasons": reasons,
    }


def actor_projection(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the information an autonomous policy may see at as-of time."""
    report = validate_scenario_fixture(fixture)
    if not report["valid"]:
        raise ValueError("SCENARIO_FIXTURE_INVALID:" + ",".join(report["reasons"]))
    public_candidates = []
    for candidate in fixture["candidate_menu"]:
        # Strict allowlist: private prediction/evaluation fields never reach
        # the autonomous policy, even if a fixture is extended later.
        item = {key: deepcopy(candidate[key]) for key in PUBLIC_CANDIDATE_KEYS if key in candidate}
        public_candidates.append(item)
    snapshot = deepcopy(fixture["public_snapshot"])
    projection = {
        "schema_version": SCHEMA_VERSION,
        "actor_view_version": "footballmind-actor-view-v1",
        "fixture_id": fixture["fixture_id"],
        "target_key": fixture["target_key"],
        "prediction_cutoff": fixture["as_of"]["prediction_cutoff"],
        "snapshot": snapshot,
        "snapshot_digest": _digest(snapshot),
        "candidate_menu": public_candidates,
    }
    report = validate_actor_projection(projection)
    if not report["valid"]:
        raise ValueError("ACTOR_PROJECTION_INVALID:" + ",".join(report["reasons"]))
    return projection


def validate_actor_projection(view: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the strict public view handed to an autonomous policy."""
    reasons: list[str] = []
    if not isinstance(view, Mapping):
        return {"valid": False, "reasons": ["actor_view_object_required"]}
    extra = set(view) - ACTOR_VIEW_KEYS
    missing = ACTOR_VIEW_KEYS - set(view)
    reasons.extend(f"actor_view_extra_field:{key}" for key in sorted(extra))
    reasons.extend(f"actor_view_missing_field:{key}" for key in sorted(missing))
    if view.get("schema_version") != SCHEMA_VERSION:
        reasons.append("actor_view_schema_mismatch")
    snapshot = view.get("snapshot")
    forbidden = {item.casefold() for item in ACTOR_FORBIDDEN_KEYS}
    def walk(value: Any, path: str = "snapshot") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                if key_text.casefold() in forbidden:
                    reasons.append(f"actor_view_private_field:{path}/{key_text}")
                walk(child, f"{path}/{key_text}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")
    if not isinstance(snapshot, Mapping):
        reasons.append("actor_snapshot_object_required")
    else:
        walk(snapshot)
        if view.get("snapshot_digest") != _digest(snapshot):
            reasons.append("actor_snapshot_digest_mismatch")
    candidates = view.get("candidate_menu")
    ids: list[str] = []
    if not isinstance(candidates, list) or not candidates:
        reasons.append("actor_candidate_menu_required")
        candidates = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            reasons.append(f"actor_candidate_{index}_object_required")
            continue
        extra_candidate = set(candidate) - PUBLIC_CANDIDATE_KEYS
        reasons.extend(f"actor_candidate_private_field:{key}" for key in sorted(extra_candidate))
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            reasons.append(f"actor_candidate_{index}_id_required")
        elif candidate_id in ids:
            reasons.append(f"actor_duplicate_candidate:{candidate_id}")
        else:
            ids.append(candidate_id)
    return {"valid": not reasons, "fixture_id": view.get("fixture_id"),
            "candidate_count": len(ids), "reasons": reasons}


def generate_scenario_fixtures() -> list[dict[str, Any]]:
    """Return fresh copies of all deterministic Round133-B fixtures."""
    fixtures = _scenario_fixtures()
    reports = [validate_scenario_fixture(item) for item in fixtures]
    invalid = [report for report in reports if not report["valid"]]
    if invalid:
        raise ValueError(f"generated scenario fixtures invalid: {invalid}")
    return deepcopy(fixtures)


def get_scenario_fixture(fixture_id: str) -> dict[str, Any]:
    requested_id = fixture_id if fixture_id.startswith("scenario-") else _opaque_fixture_id(fixture_id)
    for fixture in generate_scenario_fixtures():
        if fixture["fixture_id"] == requested_id:
            return fixture
    raise KeyError(fixture_id)


def bundle_digest(fixtures: Iterable[Mapping[str, Any]] | None = None) -> str:
    values = list(fixtures) if fixtures is not None else generate_scenario_fixtures()
    return _digest(values)


def write_fixture_bundle(path: str | Path, fixtures: Iterable[Mapping[str, Any]] | None = None) -> Path:
    """Write a deterministic JSON bundle for offline evaluator consumption."""
    values = list(fixtures) if fixtures is not None else generate_scenario_fixtures()
    reports = [validate_scenario_fixture(item) for item in values]
    if any(not report["valid"] for report in reports):
        raise ValueError("SCENARIO_FIXTURE_INVALID")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fixture_count": len(values),
        "bundle_digest": bundle_digest(values),
        "fixtures": values,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return target


__all__ = [
    "SCHEMA_VERSION", "BUNDLE_VERSION", "FORBIDDEN_PUBLIC_KEYS", "SOURCE_STATES",
    "SCENARIO_AUTHOR_HYPOTHESIS", "validate_actor_projection",
    "generate_scenario_fixtures", "get_scenario_fixture", "validate_scenario_fixture",
    "actor_projection", "bundle_digest", "write_fixture_bundle",
]
