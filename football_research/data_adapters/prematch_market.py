"""Project an archived prematch response into actor-visible numeric evidence.

The existing platform normalizer owns fixture/market semantics.  This adapter
verifies the original response bytes and projects one complete quote; it never
copies results, unrelated fixtures, or post-match fields into the actor state.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from football_spider.models.contracts import Observation
from football_spider.normalization.sources import normalize_spftll
from football_research.baselines.api import utc
from .spftll import REGULATION_SETTLEMENT


VERSION = "prematch-market-projection-v1"


def project_spftll_market_observation(
    observation: Mapping[str, Any],
    payload: bytes,
    *,
    match_id: str,
    prediction_cutoff: str,
    available_at: str | None = None,
    allow_observed_at_fallback: bool = False,
) -> dict[str, Any]:
    """Return a match row accepted by ``build_historical_asof_episode``.

    ``payload`` is the uncompressed archived body, not a re-serialized JSON
    object. Legacy archives need explicit observed-at fallback opt-in; this
    does not assert source publication or historical system ingestion time.
    """
    if not isinstance(observation, Mapping) or any(
        not observation.get(key) for key in ("id", "observed_at", "content_hash")
    ):
        raise ValueError("An explicit observation id, observed_at and content_hash are required")
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("match_id must be a nonempty source match ID")
    if type(allow_observed_at_fallback) is not bool:
        raise ValueError("allow_observed_at_fallback must be boolean")
    obs = Observation.model_validate(observation)
    if (obs.source != "spftll" or obs.http_status != 200 or obs.error_class
            or not obs.endpoint.endswith("GetSimpleMatchsAll")):
        raise ValueError("A successful spftll fixture-list observation is required")
    if not isinstance(payload, bytes) or sha256(payload).hexdigest() != obs.content_hash:
        raise ValueError("Archived payload hash mismatch")
    cutoff = utc(prediction_cutoff)
    observed = utc(obs.observed_at)
    if available_at is None and not allow_observed_at_fallback:
        raise ValueError("Missing availability; legacy observed_at fallback requires explicit opt-in")
    available = utc(available_at) if available_at is not None else observed
    if available < observed or available > cutoff or observed > cutoff:
        raise ValueError("Observation/availability is inconsistent with the prediction cutoff")

    decoded = json.loads(payload)
    groups = decoded.get("data") if isinstance(decoded, dict) else None
    if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
        raise ValueError("Expected a spftll fixture-list payload")
    target_rows = [item for group in groups for item in group.get("list", [])
                   if isinstance(item, dict) and str(item.get("matchId")) == match_id]
    # The platform normalizer deduplicates fixture identities. Reject ambiguous
    # target rows before that step, including a duplicate with missing prices.
    if len(target_rows) != 1:
        raise ValueError("Archive must contain exactly one target fixture")
    batch = normalize_spftll(obs, decoded)
    matches = [row for row in batch.matches if row.source_id == match_id]
    if len(matches) != 1:
        raise ValueError("Archive must contain exactly one target fixture")
    fixture = matches[0]
    if (fixture.status != "scheduled" or fixture.kickoff <= cutoff
            or fixture.score.home is not None or fixture.score.away is not None):
        raise ValueError("Only an unplayed fixture strictly after cutoff is eligible")
    if fixture.home.id_quality != "stable" or fixture.away.id_quality != "stable":
        raise ValueError("Stable team identities are required")
    quotes = [q for q in batch.quotes if q.source_match_id == match_id
              and q.market_type == "1x2" and q.period == "regulation" and q.line is None]
    if len(quotes) != 3 or {q.selection for q in quotes} != {"home", "draw", "away"}:
        raise ValueError("A complete unique home/draw/away quote is required")
    if any(q.settlement_rule != REGULATION_SETTLEMENT for q in quotes):
        raise ValueError("Regulation settlement semantics must be established")
    if len({(q.provider, q.bookmaker, q.source_timestamp) for q in quotes}) != 1:
        raise ValueError("All prices must come from one contemporaneous quote")
    source_id = f"spftll:observation:{obs.id}:match:{match_id}"
    quote = {
        "source_match_id": match_id,
        "provider": quotes[0].provider,
        "bookmaker": quotes[0].bookmaker,
        "market_type": "1x2", "period": "regulation", "line": None,
        "prices": {q.selection: q.decimal_price for q in quotes},
        "source_timestamp": None if quotes[0].source_timestamp is None else
            utc(quotes[0].source_timestamp).isoformat(),
        "settlement_rule": REGULATION_SETTLEMENT,
        "snapshot_id": source_id,
    }
    identity = {
        "source_match_id": match_id,
        "home_team_id": fixture.home.source_id,
        "away_team_id": fixture.away.source_id,
        "competition_id": fixture.competition.source_id,
        "kickoff": utc(fixture.kickoff).isoformat(),
    }
    visible_observation = {
        "source_id": source_id, "kind": "fixture_market_snapshot",
        "observed_at": observed.isoformat(), "content_digest": obs.content_hash,
        "values": {"fixture_identity": identity, "market_quote": quote},
    }
    if available_at is not None:
        visible_observation["available_at"] = available.isoformat()
    return {
        "match_id": match_id, "target_key": f"spftll:{match_id}", "source": "spftll",
        "competition_id": fixture.competition.source_id,
        "competition_name": fixture.competition.name,
        "home_team_id": fixture.home.source_id, "away_team_id": fixture.away.source_id,
        "home_team_name": fixture.home.name, "away_team_name": fixture.away.name,
        "kickoff": utc(fixture.kickoff).isoformat(), "prediction_cutoff": cutoff.isoformat(),
        "information_mode": "historical_reconstruction" if available_at is None else "operational",
        "source_observations": [visible_observation],
        "known": {
            "market_projection_version": VERSION,
            "source_observation_basis": "observed_at_only_not_verified_publication"
                if available_at is None else "explicit_system_availability",
            "market_settlement_basis": "existing_spftll_quote_contract",
        },
    }


__all__ = ["VERSION", "project_spftll_market_observation"]
