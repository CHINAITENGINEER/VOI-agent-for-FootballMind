"""Versioned inquiry selector construction and explicit learning exploration."""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .inquiry_exploration import UniformInquiryExploration
from .inquiry_value import DecisionRiskSelector


EXPLORATION_VERSION = "bounded-learning-inquiry-v1"


def _request_key(tool, parameters):
    if tool == "search_web":
        return (tool, parameters["query"])
    parts = urlsplit(parameters["url"])
    url = urlunsplit((parts.scheme, parts.netloc.lower(), parts.path,
                     urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True))), ""))
    return (tool, url)


class LearningExplorationSelector:
    """Spend a declared exploration budget where numerical value is unknown.

This selector has no numerical return model. Its conditional sampling is a
data-collection design, never a positive-VoI claim or global action propensity.
"""
    def __init__(self, exploration, *, max_exploration_requests):
        if not isinstance(exploration, UniformInquiryExploration):
            raise ValueError("An explicit precommitted exploration design is required")
        if type(max_exploration_requests) is not int or not 1 <= max_exploration_requests <= 6:
            raise ValueError("Exploration request cap must be 1..6")
        self.exploration, self.max_exploration_requests = exploration, max_exploration_requests

    def configuration(self):
        return {"version": EXPLORATION_VERSION, "exploration": self.exploration.configuration(),
                "max_exploration_requests": self.max_exploration_requests,
                "numeric_value_support": "none_unknown_queries_only"}

    @classmethod
    def from_configuration(cls, config):
        if config.get("version") != EXPLORATION_VERSION or config.get("numeric_value_support") != "none_unknown_queries_only":
            raise ValueError("Unsupported learning exploration contract")
        return cls(UniformInquiryExploration.from_configuration(config["exploration"]),
                   max_exploration_requests=config["max_exploration_requests"])

    def public_summary(self, run):
        spent = sum(d.get("exploration_record") is not None for d in run["decisions"])
        return {"version": EXPLORATION_VERSION, "status": "UNSUPPORTED_OPEN_TEXT",
                "expected_gain": None, "model_is_calibrated": False,
                "global_open_world_stop_proven": False, "cost_unit": "unconfigured",
                "exploration": self.exploration.public_summary(),
                "remaining_exploration_requests": max(0, self.max_exploration_requests-spent),
                "purpose": "bounded_learning_exploration_not_estimated_positive_information_value"}

    def select(self, run, response):
        raw = response["decision"]
        summary = self.public_summary(run)
        assessment = {"version": EXPLORATION_VERSION, "state": summary,
                      "global_open_world_stop_proven": False, "stop_certificate": None}
        result = {**deepcopy(raw), "selection_basis": "llm_bounded_exploration", "value_status": "UNSUPPORTED_OPEN_TEXT",
                  "expected_gain": None, "cost_in_loss_units": None, "llm_proposal": deepcopy(raw),
                  "exploration_record": None, "risk_assessment": assessment}
        if raw["kind"] != "query":
            return result
        if (summary["remaining_exploration_requests"] == 0 or
                run["budget"]["http_attempts"] >= run["budget"]["max_requests"]):
            return {**result, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "本任务的探索或请求额度已用完；保留信息价值未知，不据此推断剩余信息无用。"}
        used = {_request_key(r["tool"], r["parameters"]) for r in run["receipts"]}
        eligible, excluded = [], []
        for candidate in sorted(response["candidates"], key=lambda c: c["candidate_id"]):
            key = _request_key(candidate["tool"], candidate)
            if key in used:
                excluded.append({"candidate_id": candidate["candidate_id"], "reason": "already_attempted_or_duplicate_request"})
            else:
                eligible.append(candidate["candidate_id"])
                used.add(key)
        assessment["excluded_candidates"] = excluded
        if not eligible:
            return {**result, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "没有尚未尝试的有效探索候选；保留未知，等待新的问题或证据入口。"}
        sampled = self.exploration.select(run, response, eligible)
        return {**result, **sampled, "selection_basis": "predeclared_conditional_learning_exploration"}


def get_selector(config):
    if not config:
        return None
    version = config.get("version")
    if version == "scoped-inquiry-decision-risk-v1":
        return DecisionRiskSelector.from_configuration(config)
    if version == "model-set-inquiry-risk-v1":
        from .inquiry_ambiguity import AmbiguityRiskSelector
        return AmbiguityRiskSelector.from_configuration(config)
    if version == EXPLORATION_VERSION:
        return LearningExplorationSelector.from_configuration(config)
    raise ValueError("Unknown inquiry selector version")
