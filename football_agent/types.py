"""Private execution envelopes; domain records come from platform contracts.

These types describe tool execution, not a second match/evidence/model schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Protocol


def utc(value: datetime | str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return result.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolResult:
    status: str = "OK"
    data: Any = None
    error: Any = None
    cost: float = 0.0
    retry_after_seconds: float | None = None
    requests: int = 0
    cost_known: bool = True
    requests_known: bool = True


@dataclass(frozen=True)
class Budget:
    max_calls: int = 20
    max_queries: int = 3
    max_cost: float = 3.0
    max_elapsed_seconds: float = 60.0
    max_tool_seconds: float = 10.0
    max_requests: int = 6

    def __post_init__(self):
        for name in ("max_calls", "max_queries", "max_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_cost", "max_elapsed_seconds", "max_tool_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_tool_seconds <= 0:
            raise ValueError("max_tool_seconds must be positive")


class Tools(Protocol):
    mode: str

    def execute(self, tool: str, inputs: dict) -> ToolResult: ...
    def estimate_cost(self, tool: str, inputs: dict) -> float: ...
    def capabilities(self) -> dict: ...


@dataclass(frozen=True)
class Decision:
    action: str
    topic: str | None = None
    reason: str = ""
    action_id: str | None = None
    parameters: dict = field(default_factory=dict)


class QueryPolicy(Protocol):
    version: str

    def decide(self, run: dict) -> Decision: ...
