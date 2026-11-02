# Round 13: isolate acquired-content sensitivity in information acquisition

Written before computing round-thirteen policy results, 2026-09-26. This is an
exploratory continuation on the same frozen history, not an independent test.
The target is the user's central question: does WHAT a tool returns change the
next query/STOP decision, holding the match prior, acquired set and budget fixed,
and does retaining that content in query valuation improve realized utility?

## Frozen inputs

Reuse round11/frozen-version-v1: folds 2..6, byte-identical joint predictors and
same-version residual calibrators, source snapshots, test/validation IDs and
cutoffs. Round-twelve priors/results are references only. No model fitting,
threshold selection, new source, live request, LLM call or new independent game.
results_view and xg_view remain two projections from one cached source, costing
.02 and .03, one request each. Fixed lambda and reporting alpha .5, budgets
0/.02/.03/.05, maximum two requests, positive rounded-12-decimal net value,
action-ID tie breaking. Predictor probabilities always update from REAL revealed
returns in executed policies. Provider failure remains outside the fitted model.

## One new control: content-marginal-mean

Let p0 be the same frozen match prior, S the already acquired action set and
g_hat(a; p0, z_S) the existing same-version calibrated mean gain, evaluated using
the frozen conditional model EVSI and the same calibrated residual cells.
For S empty, retain EXACTLY the original numerical mean gains and first decision.
For S containing one action b, replace the valuation of the remaining action a by

    g_bar(a; p0, {b}) = sum_z P_psi(Z_b=z | p0) * g_hat(a; p0, Z_b=z).

Enumerate all positive-probability categories, including the missing-summary
category, under the fixed joint model. Average AFTER applying the original
nonlinear EVSI-bin residual correction to each branch. Applying a calibrator to
averaged EVSI is a different estimator and is excluded. The averaging weights
come from the free prior joint cube, never from the actual observed category,
posterior, remaining hidden return, validation result or test outcome. There are
at most ten branches. This is an exact finite sum, not sampling or new training.

The control retains the match prior and acquisition stage while erasing actual
return content from QUERY VALUATION ONLY. Initial actions must equal the original
policy. Thereafter actual outcome probabilities still use real acquired returns.
If any positive-weight branch lacks a supported mean, mark the candidate
unsupported and STOP; do not silently discard or renormalize such branches.
With both tools acquired there is no candidate. No LCB policy or invented
aggregate uncertainty interval: lower_gain and standard_error are null for the
wrapper. Native branch estimates may retain their descriptive audit statistics.

Serialize an explicit estimator kind content-marginal-gain-v1, binding exact
joint and source residual-calibrator hashes/versions. Native model_evsi and
evsi_bin are null at the wrapper level; store marginal_model_evsi separately,
plus branch weights, conditional EVSIs, gains, support and original source
versions. Runtime hypothetical envelopes are still rejected. Planning-only
category branches stay internal mathematical objects, never tool receipts or
observed evidence. The wrapper supports the existing mean-only plan interface.

## Two complementary evaluations

1. Executed policy ablation. Run only the new control: 200*4=800 test episodes and
213*4=852 validation episodes. Reuse frozen conditional, stage/tool prior and old
reference episodes. Primary contrast: frozen-version-mean minus
content-marginal-mean utility loss at budget .05, paired over the same 200 games.
Report first-action equality, 0/1/2 queries, cost, probabilities, realized gains
and STOP disagreements. Initial STOP games remain STOP in both policies. Smaller
budgets offer no affordable second query and must have identical paths, costs
and predictions. Validation is descriptive only; no policy selection from it.
Use existing equal-game, sorted UTC-week bootstrap, 2,000 draws, seed1729.

2. Within-match categorical branch diagnostic. For every test and validation
match and BOTH possible first tools, enumerate every fitted positive-mass return
category at the fixed p0. Compare original conditional next-action/STOP with the
single content-marginal decision at remaining budget .05-cost(first), requests1.
Record branch probability, posterior, remaining model EVSI, calibrated mean,
net gain and decision; summarize within-match ranges, policy switches, and
probability-weighted disagreement. Also project the actually reconstructed
category into the table and check it agrees with frozen actual-state diagnostics.
Report separately (a) all 826 match/first-tool contexts, (b) test 400 contexts,
(c) contexts reached after an actual first query by the frozen policy. Unvisited
histories and hypothetical categories are not extra matches, actual queries or
observed outcomes. Never pair an invented return with the original match outcome
and call it a causal outcome. These tables establish sensitivity INSIDE the
fitted model, not the truth of alternative football events or real decision gain.

Primary confidence intervals exclude fitting, repeated research, multiplicity
and distribution-shift uncertainty. A null interval is not evidence of equality.
The new control still depends on the possibly misspecified return model; the
ablation estimates a fixed-policy difference on reused history, not a general
causal benefit of information or proof of open-environment transfer.

## Execution, independent checks and Agent

Root independently reconstructs p0-weighted joint cubes from frozen training
records and computes branch EVSIs/calibrated means/weighted values; verify all
new trajectories, aggregate metrics and common-state/branch decisions. Tests
must detect posterior-dependent weights, averaging EVSI before calibration,
dropping unsupported/missing branches, hidden-payload or outcome leakage,
wrong acquisition set, invalid artifacts and hypothetical tokens at runtime.

Run the new Agent control on each fold's earliest scheduled test match, sorting
by kickoff and match ID: five fixed new workflows. Reuse the five already frozen
corresponding conditional workflows/cards. Audit value snapshots, selected
actions, receipts, posterior updates, cost and STOP; record explicit wrapper
kind and content-erasure scope in cards. Internal hypothetical branch audit is
labelled model-only; it never counts as acquisition. No favorable-case selection.

Two existing work agents can implement the research estimator/runner and the
Agent adapter in parallel; root owns independent verification, branch analysis
and final acceptance. Add new round13 files; keep previous code/data/results
immutable except two cumulative status documents after final verification.
Complete by running relevant regressions, inspecting the scientific figure,
documenting negative results as readily as positive ones, and freezing hashes.
