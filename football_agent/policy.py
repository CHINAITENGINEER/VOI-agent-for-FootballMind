"""Replaceable acquisition policy. Fixed rules are an engineering baseline."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from .types import Decision


class FixedOrderPolicy:
    version = "fixed-order-v1"

    def __init__(self, topics=("odds", "lineups", "injuries")):
        if not topics or any(not isinstance(t, str) or not t.strip() for t in topics):
            raise ValueError("Query topics must be nonempty strings")
        self.topics = tuple(dict.fromkeys(topics))
        self.configuration = {"topics": list(self.topics)}

    def decide(self, run: dict) -> Decision:
        if run["budget"]["remaining_queries"] <= 0:
            return Decision("stop", reason="QUERY_BUDGET_EXHAUSTED")
        queried = run.get("queried_topics", [])
        for topic in self.topics:
            if topic not in queried:
                return Decision("query", topic, "FIXED_QUERY_ORDER")
        return Decision("stop", reason="QUERY_PLAN_COMPLETE")


class CallbackPolicy:
    """VOI/learned/Agent integration seam; execution still uses the same guardrails.

    This callback does not imply an LLM has been configured or tested. Callers
    must account separately for any planning API costs in their tool adapter.
    """
    def __init__(self, callback: Callable[[dict], Decision], version="callback-v1"):
        self.callback, self.version = callback, version

    def decide(self, run: dict) -> Decision:
        result = self.callback(deepcopy(run))
        if not isinstance(result, Decision) or result.action not in {"query", "wait", "stop", "defer"}:
            raise ValueError("Policy must return a supported Decision")
        return result


class ToolPlanningPolicy:
    """Agent planner integration that runs as an audited, budgeted tool call.

    The tools adapter must implement ``plan_next_action`` and estimate its cost.
    No external planner is enabled by this class alone.
    """
    version = "tool-planning-v1"
    tool_name = "plan_next_action"

    @staticmethod
    def decode(data):
        if not isinstance(data, dict) or set(data) - {"action", "topic", "reason"}:
            raise ValueError("Invalid planner output")
        result = Decision(**data)
        if result.action not in {"query", "wait", "stop", "defer"}:
            raise ValueError("Unsupported planner action")
        if result.action == "query" and (not isinstance(result.topic, str) or not result.topic.strip()):
            raise ValueError("Planner query requires a topic")
        return result
