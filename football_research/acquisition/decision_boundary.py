"""Exact accounting for acquisition decisions, with no fitted parameters.

Realized gains belong to offline diagnostics. They are not expected values or
inputs to the policy. Multiple transitions from one match remain dependent.
"""
from __future__ import annotations

from bisect import bisect_right
import math


MARGIN_EDGES = (-.05, -.02, -.01, 0., .01, .02, .05)
TERMS = ("decision_margin", "cost_weight_difference", "mean_minus_decision", "calibration_residual")


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+" must be a finite real number")
    return float(value)


def margin_bucket(value):
    return bisect_right(MARGIN_EDGES, finite(value,"margin"))


def decompose_query(*, gain, mean_estimate, decision_estimate, cost, policy_lambda, utility_weight):
    values = {k:finite(v,k) for k,v in locals().items()}
    if min(values[k] for k in ("cost","policy_lambda","utility_weight")) < 0:
        raise ValueError("Costs and penalty weights must be nonnegative")
    g,h,m,c,lam,alpha = (values[k] for k in ("gain","mean_estimate","decision_estimate","cost","policy_lambda","utility_weight"))
    terms = {"decision_margin":m-lam*c,"cost_weight_difference":(lam-alpha)*c,
             "mean_minus_decision":h-m,"calibration_residual":g-h}
    actual = g-alpha*c
    if not math.isclose(math.fsum(terms.values()),actual,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError("Acquisition accounting identity failed")
    return {**terms,"actual_gain":g,"actual_utility_gain":actual,
            "reported_estimated_net_gain":h-alpha*c,"margin_bucket":margin_bucket(terms["decision_margin"])}


def sum_episode(queries, *, initial_loss, final_loss, cost, utility_weight):
    for name,value in (("initial_loss",initial_loss),("final_loss",final_loss),("cost",cost),("utility_weight",utility_weight)):
        finite(value,name)
    if cost < 0 or utility_weight < 0:
        raise ValueError("Negative episode cost/weight")
    gain = math.fsum(q["actual_gain"] for q in queries)
    utility = math.fsum(q["actual_utility_gain"] for q in queries)
    if not math.isclose(gain,initial_loss-final_loss,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("Query gains do not telescope to the episode")
    if not math.isclose(utility,initial_loss-final_loss-utility_weight*cost,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("Episode utility accounting differs")
    return {**{k:math.fsum(q[k] for q in queries) for k in TERMS},"actual_gain":gain,
            "actual_utility_gain":utility,"queries":len(queries)}


def summarize_records(records):
    """A descriptive transition average, explicitly not independent match n."""
    fields = (*TERMS,"actual_gain","actual_utility_gain","reported_estimated_net_gain")
    return {"records":len(records),"distinct_matches":len({r["match_id"] for r in records}),
            **{k:math.fsum(r[k] for r in records)/len(records) if records else None for k in fields},
            "scope":"descriptive transition averages; records sharing a match are dependent"}
