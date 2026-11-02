"""Explicit synthetic fixtures for workflow verification, never real forecasts."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

from .adapters import digest, utc
from .types import ToolResult


TEST_MATCH_ID = "test:match-001"
TEST_CUTOFF = "2026-09-20T11:30:00+00:00"
SCENARIOS = ("normal", "empty", "timeout", "rate_limited", "duplicate", "stale", "conflict", "budget", "model_unavailable")


class SyntheticTools:
    """Deterministic stress-test tools with a clearly labelled test model.

    All evidence predates the fixed cutoff; refresh progressively reveals local
    fixtures. This is tool orchestration verification, not an empirical backtest.
    """
    mode = "test"

    def __init__(self, scenario="normal"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown synthetic scenario {scenario!r}")
        self.scenario = scenario
        self.refresh_count = 0

    def capabilities(self):
        return {"mode": "test", "label": "测试模式 / 合成压力测试", "scenario": self.scenario,
                "tools": {name: "SYNTHETIC" for name in ("query_matches", "read_evidence", "get_features", "predict", "refresh_information")},
                "model": "synthetic-test-model-v1", "external_requests_enabled": False,
                "default_match_id": TEST_MATCH_ID, "default_cutoff": TEST_CUTOFF}

    def estimate_cost(self, tool, inputs):
        return (100.0 if self.scenario == "budget" else 0.25) if tool == "refresh_information" else 0.0

    def execute(self, tool, inputs):
        handler = getattr(self, "_" + tool, None)
        if handler is None or tool.startswith("_"):
            return ToolResult(status="FAILED", error=f"Unknown synthetic tool: {tool}")
        return handler(dict(inputs))

    @staticmethod
    def _match():
        return {"match_id": TEST_MATCH_ID, "home": "测试主队", "away": "测试客队", "source": "synthetic",
                "kickoff": "2026-09-20T12:00:00+00:00", "fixture_available_at": "2026-09-20T10:00:00+00:00",
                "competition": "合成压力测试", "competition_id": "test:competition", "home_team_id": "test:home",
                "away_team_id": "test:away", "status": "scheduled", "information_mode": "synthetic_test",
                "input_manifest": ["synthetic://fixture/match-001"]}

    def _query_matches(self, inputs):
        match = self._match()
        cutoff = utc(inputs.get("prediction_cutoff") or inputs.get("cutoff") or TEST_CUTOFF)
        if (inputs.get("match_id") and inputs["match_id"] != TEST_MATCH_ID) or utc(match["fixture_available_at"]) > cutoff:
            return ToolResult(status="NO_DATA", data=[], error="No synthetic fixture visible at cutoff")
        return ToolResult(data=[match])

    def _item(self, version=1):
        content = {"label": "合成测试证据", "test_value": version,
                   "description": "用于验证工具调用、预算与回放；不是实际球队信息"}
        observed = utc(TEST_CUTOFF) - timedelta(minutes=30 if version == 1 else 15)
        item = {"evidence_id": f"synthetic-evidence-{version}", "version": digest(content),
                "match_id": TEST_MATCH_ID, "source": "synthetic", "kind": "odds", "topic": "odds",
                "observed_at": observed.isoformat(), "available_at": observed.isoformat(),
                "content_hash": digest(content), "content": content,
                "raw_ref": f"synthetic://evidence/{version}", "information_mode": "synthetic_test"}
        if self.scenario == "stale":
            item["expires_at"] = (utc(TEST_CUTOFF) - timedelta(minutes=5)).isoformat()
        if self.scenario == "conflict":
            item["conflict_key"] = "synthetic:lineup:captain"
            item["observed_at"] = (utc(TEST_CUTOFF) - timedelta(minutes=30)).isoformat()
            item["available_at"] = item["observed_at"]
        return item

    @staticmethod
    def _visible(items, inputs):
        cutoff = utc(inputs.get("prediction_cutoff") or TEST_CUTOFF)
        return [deepcopy(item) for item in items if item["match_id"] == inputs.get("match_id", TEST_MATCH_ID)
                and utc(item["available_at"]) <= cutoff]

    def _read_evidence(self, inputs):
        items = [] if self.scenario == "empty" else [self._item()]
        if self.scenario == "conflict":
            items.append(self._item(2))
        visible = self._visible(items, inputs)
        return ToolResult(status="OK" if visible else "NO_DATA", data=visible,
                          error=None if visible else "Synthetic empty evidence response")

    def _get_features(self, inputs):
        if not inputs.get("evidence"):
            return ToolResult(status="NO_DATA", error="No accepted synthetic evidence")
        return ToolResult(data={"match_id": inputs["match_id"], "prediction_cutoff": inputs["prediction_cutoff"],
                               "feature_version": "synthetic-feature-v1", "information_mode": "synthetic_test",
                               "input_manifest": [{"evidence_id": item["evidence_id"], "version": item["version"]}
                                                  for item in inputs["evidence"]],
                               "test_value": max(item["content"]["test_value"] for item in inputs["evidence"])})

    def _predict(self, inputs):
        if self.scenario == "model_unavailable":
            return ToolResult(status="MODEL_UNAVAILABLE", error="Synthetic injected model outage")
        features = inputs.get("features") or {}
        if "test_value" not in features:
            return ToolResult(status="NO_DATA", error="Synthetic model requires synthetic features")
        home = 0.45 if features["test_value"] == 1 else 0.47
        return ToolResult(data={"match_id": inputs["match_id"], "prediction_cutoff": inputs["prediction_cutoff"],
                               "home_win_probability": home, "draw_probability": 0.28, "away_win_probability": 1 - home - 0.28,
                               "model_version": "synthetic-test-model-v1", "dataset_version": "synthetic-stress-test-v1",
                               "information_mode": "synthetic_test", "test_mode": True,
                               "label": "测试模式：数值来自测试替身，不是实际比赛胜率",
                               "input_manifest": {"feature_input": features["input_manifest"]}})

    def _refresh_information(self, inputs):
        self.refresh_count += 1
        cost = self.estimate_cost("refresh_information", inputs)
        if self.scenario == "timeout":
            return ToolResult(status="TIMEOUT", error="Synthetic source timeout", cost=cost)
        if self.scenario == "rate_limited":
            return ToolResult(status="RATE_LIMITED", error="Synthetic HTTP 429", cost=cost, retry_after_seconds=30)
        if self.scenario == "empty":
            return ToolResult(status="NO_DATA", data={"evidence": []}, error="Synthetic empty query result", cost=cost)
        items = [self._item(1 if self.scenario == "duplicate" else 2)]
        return ToolResult(data={"evidence": self._visible(items, inputs), "synthetic": True,
                                "query_topic": inputs.get("topic")}, cost=cost)
