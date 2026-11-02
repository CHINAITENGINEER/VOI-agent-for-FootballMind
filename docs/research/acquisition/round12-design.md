# Round 12: does current public evidence improve autonomous information choice?

Written before running round-twelve comparisons, 2026-09-26. Reuse round-eleven
frozen artifacts and previously explored history. No new independent evidence is
claimed. The question is whether state-conditional tool/STOP decisions improve
upon past average tool gains, with the same predictor and acquisition costs.

## Frozen inputs and targets

Use round11/frozen-version-v1, target folds 2..6, each exact reference predictor
from original fold k-2, same-version calibration samples, target validation/test
IDs, information modes, catalog and timestamps. Do not regenerate or refit the
joint outcome predictor or round-eleven residual calibrator. Source outcome
availability, training exclusions and no-inner-tail protocol remain unchanged.

The gain target is the same fixed-predictor single-query log-loss reduction.
Costs remain simulated results_view=.02 and xg_view=.03, one request per query.
Execution lambda and reporting alpha stay .5, budgets 0/.02/.03/.05, max 2
requests. Positive rounded-12-decimal net gain wins; tie by action ID, STOP at
nonpositive best gain. No validation selection or parameter search.

## Two necessary simpler controls

1. tool-prior-mean: action identity only. For each calibration match/action,
   average its two gains (empty state and the other tool already acquired),
   then average those per-match values. This explicit uniform weighting of the
   two recorded states is a coarse reference, not a learned occupancy model.
   Runtime value is constant by fold and action, even after another return.

2. stage-prior-mean: action identity and acquired-action set only. Average past
   gains for that action/mask, one observation per calibration match. No current
   probabilities, EVSI, team identity, fixture date, or acquired return contents
   enter the numerical value. Source-missing is still an acquired response.

Both use all the same complete four-transition same-version records and match
IDs. Unique-match support threshold remains 12. Tool-prior averages within match
before measuring support or variance. Descriptive week-cluster standard errors
may be retained for audit, but only mean policies are evaluated in this round.
No new LCB policy is introduced. Serialize an explicit new estimator kind;
model_evsi must be null for these controls, not a made-up zero or residual label.

Runtime identity/time/availability/history/budget checks are still enforced.
The joint predictor still consumes actual acquired contents to update outcome
probabilities in every policy. Erasure concerns the query-value estimator only.
Tests must demonstrate that changing admissible public numeric contents or team
identity leaves these baseline values unchanged, while a changed acquired set
can change stage-prior value. No current hidden return or outcome enters choice.

## Comparisons and statistics

Primary: round-eleven frozen-version-mean versus stage-prior-mean, budget .05,
using paired fixed utility loss. This asks whether the implemented conditional
estimator beats a mask-aware average-gain estimator. It does not identify a pure
causal benefit of conditioning separately from estimator bias/variance.

Secondary: conditional versus tool-prior, stage-prior versus tool-prior, both
new baselines versus base-only, and original reference policies. Retain all four
budgets. Read frozen round-eleven reference episodes without rerunning them.
Expected new episodes: 200*4*2=1,600 test and 213*4*2=1,704 validation. Original
validation remains descriptive and is not used to choose a winning baseline.

Use equal match weights, existing sorted UTC-week bootstrap 2,000 draws/seed1729.
Repeated budgets, dependent transitions and already explored games do not create
independent samples. Intervals omit fitting, repeated-research and multiplicity
uncertainty; do not equate crossing zero with equivalence.

Report per-fold tool/STOP distributions, 0/1/2 queries, actual cost, query gains
and post-return decisions. Verify within-fold value invariance for the erased
controls. Compare initial actions and remaining-tool decisions on the SAME
public states from the complete frozen transition dataset, so differing policy
paths are not mistaken for a matched state contrast. Report how many net-value
signs change after the same observed return for each policy; average-state
changes and sensitivity to actual contents are distinct claims.

## Actual Agent and completion

Run both new controls on each fold's earliest scheduled test match, sorting by
kickoff then ID (10 new workflows). Reuse the 5 already audited corresponding
same-version-mean decision cards for the conditional comparison. No favorable
case selection. Build new cards through the existing event-snapshot builder;
show estimator kind and state-erasure mode, and null EVSI. Confirm selected
actions, receipts, probability updates, STOP and cost equal research traces.

Keep old code/artifacts frozen. Add only the estimator, runner, required Agent
adapter and independent verification/analysis/tests. Two existing work agents
may implement research and Agent integration in parallel; root independently
checks the numerical and behavioral contrast. No new scraper, HTTP request,
LLM call, schedule, user-owned chat or deployment is part of this experiment.

Complete by retaining all results, including negative findings, running relevant
regressions, and freezing report/artifact hashes. Qualification, rotation,
lineups and tactical effects remain future evidence domains; this round tests
autonomous acquisition in the existing two-view historical environment.
