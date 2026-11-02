# Round 14: calibrated continuation value and adaptive information acquisition

Frozen before computing round-fourteen planning/policy results, 2026-09-26.
The question is whether a first query should purchase an opportunity to decide
what to acquire next after seeing its actual contents. This is exploratory work
on already explored history, not independent effectiveness confirmation.

## Frozen environment and estimands

Keep original round11/frozen-version-v1 folds 2..6, exact joint predictors and
same-version residual calibrators, calibration/validation/test IDs, snapshots,
timelines and policies. Read original round11, round12 and round13 episodes as
references without rerunning. No new fit, source, HTTP/LLM use or parameter search.
Two historical views of one cached source: results=.02, xg=.03, one request each;
budgets 0/.02/.03/.05, at most two requests; execution/reporting coefficient .5.
Numerical query/STOP selection uses rounded-12-decimal positive net scores and
action-ID ties. Outcome prediction always uses actual revealed historical returns.

The old calibrated g_hat is an empirical update-gain estimate. Combining it with
the frozen return distribution is an approximate planning reward model, not a
coherent Bayes-risk theorem or a guarantee of real-world VoI. Preserve this label.
The old raw-EVSI two-step planner (round8) is not the new intervention.

## Two new policies, no tuning

For free-context state s, first action a and remaining action b, write
n_a = g_hat(a,s)-lambda*c_a. For every possible first return z, use the frozen
joint probability w_z=P_psi(Z_a=z|s), and let
d_z=g_hat(b,s+z)-lambda*c_b, with the ORIGINAL calibration applied to each branch.

1. calibrated-adaptive-two-step:

   Q_adaptive(a,s) = n_a + sum_z w_z * d_z * 1[round(d_z,12)>0].

   Thus each future branch may STOP independently. Select the first action with
   largest positive rounded Q, or STOP. After a REAL first return, recompute the
   original conditional one-step mean decision. No hypothetical token is executed.

2. calibrated-commit-two-step:

   Q_commit(a,s) = n_a + d_bar * 1[round(d_bar,12)>0],
   d_bar=sum_z w_z*g_hat(b,s+z)-lambda*c_b.

   The continuation is decided without using the actual first-return content.
   After the real first return, use the frozen round13 content-marginal mean
   rule, which is identical to this continuation decision in the unchanged
   two-view environment. Forecasts still use the real contents. This is an
   open-loop continuation control, not a command to ignore changed availability,
   failures or physical budgets at runtime.

With no affordable/legal second action, both Qs reduce to the original current
one-step value. If a positive-mass future branch has no supported mean, disable
the continuation bonus for BOTH modes, retaining null estimates and an explicit
unsupported-continuation reason; never drop or reweight that branch. A supported
first query can still run for its immediate gain. If the current immediate gain
is unsupported, the first action is unsupported. With both actions acquired STOP.
Keep missing-summary branches. Provider failures remain outside the model.

Use physical budgets for every possible branch and all resource units. Expected
future cost is only a planning quantity, never an immediate charge or permission
to exceed a per-branch limit. Respect unavailable/already-attempted actions and
duplicate evidence, at the root and in continuation. A first action with negative
immediate net may be selected only if the supported continuation makes total Q
positive. Report this explicitly rather than blocking it with a one-step gate.

## Numerical and audit contract

New explicit kind calibrated-continuation-planner-v1, modes adaptive/commit,
mean-only, maximum horizon two; bind exact joint and source-calibrator hashes.
No new uncertainty interval or LCB policy. Save immediate_estimate separately.
Expose immediate_expected_gain, immediate_cost_penalty, immediate_net_gain,
continuation_bonus, expected_continuation_gain, expected_continuation_cost,
expected_gain (= expected SUM of calibrated gains along this conditional plan),
cost_penalty (= expected TOTAL cost converted to loss units), mean_net_gain/Q,
selection_score, and per-return probability/gain/continuation decision audit.
Top-level standard_error/lower_gain/model_evsi are null; source estimates keep
their own descriptive statistics. Plan gain must never be labelled single-query
gain, and projected future spending must never be recorded as actual spending.

For a matched initial action the adaptive bonus minus commit bonus is the value
of choosing QUERY/STOP separately by return under this fitted score model. Report
it as a surrogate option gap, not measured true information benefit. Away from
rounding boundaries it is E[max(0,d)]-max(0,E[d]); near zero allow 1e-12 numerical
tolerance instead of claiming an exact inequality from rounded decisions.

## Evaluation and independent verification

Run only the two new policies: 200*4*2=1600 test and 213*4*2=1704 validation
episodes. Primary: adaptive minus commit utility loss at budget .05. Secondary:
adaptive versus frozen-version-mean; commit versus content-marginal-mean;
each versus base-only and round12 stage/tool means. All budgets remain visible.
Validation is descriptive only, with no selection. Average seeds within match,
equal weight matches, sorted UTC-week bootstrap 2000/seed1729 as before. Intervals
exclude fitting, repeated research, multiplicity and temporal-shift uncertainty.

Report every initial candidate's immediate net, adaptive/commit bonus, option gap,
total score, expected requests/cost and branch-specific continuation. Compare
the same initial states across both modes; show which first actions and initial
STOPs differ from one-step references. Report realized paths with 0/1/2 queries,
actual costs, probability updates, and how many negative-immediate-net queries
were justified by modeled continuation. Validate all low-budget new paths equal
their corresponding one-step reference. Do not optimize a threshold for more
second queries. Record all negative or zero findings.

Use the full fixed-prior branch analysis to separate opportunity under the model,
states actually reached and decisions under the actual observed category. Never
attach a real match outcome to an invented return to score a counterfactual.
For actual new paths only, decompose realized first/second log-loss changes and
costs; verify telescoping to base minus final utility. This retrospective label
is never a runtime feature. Aggregate branch counts are not independent games.

Root independently rebuilds joint cubes and the old residual correction from
training records, computes both continuation policies without the production
planner, and checks every new trajectory, summary and candidate. Regression
tests cover negative immediate but positive continuation, adaptive-vs-commit
gap, per-branch budgets, one remaining query, post-return content dependence,
missing/unsupported branches, hypothetical boundary, artifact binding and costs.

## Agent, scope and completion

Run both new modes on each fold's earliest scheduled test match (kickoff, ID):
10 new historical Agent workflows. Reuse corresponding frozen one-step cards
from round11 and round13; no outcome/gain-driven selection. Cards show immediate
gain, continuation contribution, total planned gain/cost and actual receipt
separately. Root checks event/blob integrity and independently recomputes values.
Old default factories, implementations, runs and model artifacts stay frozen.

Two existing work agents may implement research and Agent integration in parallel;
root owns design, independent verification, analysis and final acceptance. Add
round14 files only, except two cumulative status notes after prior hashes pass.
No new user-owned chat or scheduler. Complete relevant regressions, inspect the
scientific figure, and freeze result hashes. This round tests autonomous next-
information selection; it does not implement new qualification/lineup evidence.
