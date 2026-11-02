"""Validate source assertions before they can be considered by f or g.

This contract deliberately sits before the prediction pipeline. It validates
that a human-readable assertion is tied to an exact body span and that market
or team-context statements are not silently promoted to match facts. It emits
no prediction, gain, propensity, or training row and has no network path.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import math


SCHEMA = "footballmind-offline-source-assertion-handoff-v1"
FORBIDDEN_TRAINING_KEYS = {
    "prediction", "prediction_before", "prediction_after", "gain", "loss",
    "propensity", "training_eligible", "g_training_eligible",
}


def _time(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _quote_span(body, evidence):
    if not isinstance(evidence, dict):
        raise ValueError("Evidence span must be an object")
    quote = evidence.get("quote")
    start, end = evidence.get("start"), evidence.get("end")
    if not isinstance(quote, str) or not quote:
        raise ValueError("Evidence quote must be nonempty")
    if evidence.get("offset_unit") != "unicode_codepoints":
        raise ValueError("Only unicode_codepoints offsets are supported")
    if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(body)):
        raise ValueError("Evidence offsets are outside the body")
    if body[start:end] != quote:
        raise ValueError("Evidence quote does not match the body at its offsets")


def _walk_forbidden(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_TRAINING_KEYS and child is not None:
                raise ValueError(f"Training field {path}.{key} is populated")
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def _validate_assertion(row, assertion):
    if not isinstance(assertion, dict):
        raise ValueError("Assertion must be an object")
    required = {"assertion_id", "type", "provenance", "evidence"}
    if not required <= assertion.keys():
        raise ValueError("Assertion is missing its identity, type, provenance, or evidence")
    if assertion["provenance"] != "source_assertion_not_independent_truth":
        raise ValueError("Only source assertions with explicit non-truth provenance are accepted")
    if not isinstance(assertion["evidence"], dict):
        raise ValueError("Assertion evidence is required")
    _quote_span(row["_body_text"], assertion["evidence"])
    kind = assertion["type"]
    if kind == "third_party_probability":
        values = assertion.get("values")
        if (not isinstance(values, dict) or set(values) != {"home", "draw", "away"}
                or any(type(values[k]) not in (int, float) or not math.isfinite(values[k])
                       or not 0 <= values[k] <= 1 for k in values)
                or not math.isclose(sum(values.values()), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError("Third-party probabilities must be finite, bounded, and normalized")
        if assertion.get("source_model_calibration") != "unknown":
            raise ValueError("Uncalibrated third-party probabilities must remain marked unknown")
        if assertion.get("probability_quote_time") is not None:
            _time(assertion["probability_quote_time"], "probability_quote_time")
        if assertion.get("not_shadow_f_prediction") is not True:
            raise ValueError("Third-party probability cannot be treated as the shadow f forecast")
    elif kind == "reported_market_odds":
        odds = assertion.get("decimal_odds")
        if (not isinstance(odds, dict) or not odds
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 1
                       for value in odds.values())):
            raise ValueError("Reported decimal odds must be finite and greater than one")
        if assertion.get("direct_bookmaker_observation") is not False:
            raise ValueError("Article-reported odds are not direct bookmaker observations")
        if assertion.get("quote_time") is not None:
            _time(assertion["quote_time"], "quote_time")
    elif kind in {"injury", "availability_for_other_match", "recovery_training", "return_uncertainty"}:
        if assertion.get("scope") != "team_context":
            raise ValueError("Indirect injury/background assertions must remain team_context")
        if assertion.get("this_match_assertion") is True:
            raise ValueError("Indirect team context cannot be promoted to this_match")
        if assertion.get("target_fixture_status") != "unknown":
            raise ValueError("Unknown target applicability must remain explicit")
    else:
        raise ValueError(f"Unsupported source assertion type: {kind}")


def validate_record(row, *, body_root):
    if not isinstance(row, dict):
        raise ValueError("Source handoff record must be an object")
    row = deepcopy(row)
    if row.get("schema") != SCHEMA:
        raise ValueError("Unknown source assertion handoff schema")
    _walk_forbidden(row)
    for key in ("article_id", "source_url", "body_path", "inherited_body_sha256",
                "target_physical_match_id", "target_kickoff", "target_prediction_cutoff"):
        if not row.get(key):
            raise ValueError(f"Missing required handoff field: {key}")
    if _time(row['target_prediction_cutoff'], 'target_prediction_cutoff') >= _time(row['target_kickoff'], 'target_kickoff'):
        raise ValueError('Prediction cutoff must strictly precede kickoff')
    if row.get("prospective_training_eligibility_claimed") is not False:
        raise ValueError("This handoff must not claim prospective training eligibility")
    if row.get("retain_in_research_pool") is not True:
        raise ValueError("Source record must remain auditable in the research pool")
    if body_root is not None:
        body_path = Path(row["body_path"])
        root = Path(body_root).resolve()
        if not body_path.is_absolute():
            body_path = root / body_path
        body_path = body_path.resolve()
        if not body_path.is_relative_to(root):
            raise ValueError("Body path escapes the declared handoff root")
        # read_bytes().decode() deliberately preserves CRLF: the handoff
        # offsets are measured against the exact UTF-8 body bytes, not Python's
        # universal-newline text view.
        body = body_path.read_bytes().decode("utf-8")
        row["_body_text"] = body
    else:
        raise ValueError('A body root is required; evidence validation cannot be skipped')
    if not isinstance(row.get("assertions"), list) or not row["assertions"]:
        raise ValueError("At least one source assertion is required")
    if row["_body_text"] is not None:
        for assertion in row["assertions"]:
            _validate_assertion(row, assertion)
    row.pop("_body_text", None)
    return row


def validate_jsonl(path, *, body_root):
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(validate_record(json.loads(line), body_root=body_root))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid handoff row at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("The handoff must contain at least one row")
    if len({row["article_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate article_id in handoff")
    return rows
