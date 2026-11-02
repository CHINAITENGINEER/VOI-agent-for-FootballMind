# Round 10: decision-boundary and support diagnosis

Recorded before running the new fixed-cost contrast, 2026-09-26.

This round uses the previously explored round-nine artifacts. It is a bounded
diagnostic iteration, not an independent confirmation or a search for a winning
policy. No model, residual cell, data source, uncertainty multiplier, or EVSI bin
is refitted. Existing artifacts remain frozen.

## One prespecified contrast

Keep the round-nine predictor, calibrator, physical budgets (0, .02, .03, .05),
request ceiling, temporal folds and information unchanged. Execute the existing
mean and conservative one-step rules with lambda_money equal to the previously
fixed reporting utility weight alpha=.5. Compare with the original
validation-selected-lambda mean/LCB policies and zero acquisition.

Alpha is read from the frozen round-nine design, not selected from these new
test results. No other fixed lambda is run. This contrast asks whether the
cost conversion and validation choice explain behavior, not whether .5 is an
optimal deployment penalty. Lambda tuning can legitimately compensate for
miscalibration; lambda differing from alpha is not an implementation bug.

Recompute the fixed-alpha policy on the original validation partitions, check
its utility against the already recorded lambda=.5 validation candidate, and
replay the same 290 old test games. Report both validation and test differences;
small validation samples and temporal changes are possible explanations, not
established causes.

## Exact accounting of chosen queries

For each actual chosen query, c is simulated cost, g_hat is the mean estimate,
m is the estimate used for the decision (mean or descriptive lower summary),
g is the realized fixed-predictor update gain, and lambda is the policy penalty.

    g - alpha*c = (m - lambda*c) + (lambda-alpha)*c
                  + (g_hat-m) + (g-g_hat)

Verify this identity per query and its telescoping sum per episode. Aggregate
episode contributions with zeros for stopped matches. The decomposition is
algebraic, not a causal attribution. Repeated budgets and multiple transitions
from the same match never count as additional independent matches.

Decision-margin diagnostic boundaries are fixed now at
[-.05, -.02, -.01, 0, .01, .02, .05], with equality in the right bucket. Keep
action, acquired-set, fold, risk rule, and support metadata. Report selected
queries separately from all affordable candidates at the initial public state.
Actual gains enter offline evaluation only; they never select an action.
No hindsight-best policy, oracle regret, or single-result information-value
claim is reported.

Use the established equal-match, sorted UTC-week bootstrap for strategy
differences (2,000 resamples, seed1729). Intervals remain descriptive and omit
research-selection, model-fit and multiplicity uncertainty. No new margin or
bin is chosen after seeing results.

## Independent support investigation

Using the frozen calibration fit population, compare fixed EVSI cells across
fit, diagnostic tail, original validation and test. Range and 5/95 quantile
checks use only public features. Compare prior probabilities, entropy, nearest
fit-prior distance and the training sizes/times of the OOF and deployed joint
models. Use leave-one-match-out distances within fit. Inspect the original
calibrated policies' selected queries separately. No new threshold gates are
trained from test gains. Model size and time change together: describe their
joint differences without a causal drift claim.

## Prospective boundary

Add a small pure metadata contract validator to bind predictor, calibrator,
catalog, gain target, cost units, data-availability cutoffs, domain, selection
rule and deployment window. Historical reconstruction and simulated frozen
projections retain their retrospective classification. A valid metadata
contract is not proof of source truth or completed live evaluation. No live
deployment, crawler requests, task scheduling or new source onboarding occurs
in this round.

## Completion criteria

New fixed-cost trajectories independently reproduce frozen-model predictions,
calibrated candidate values, actions, costs and stopping. Validation .5 scores
equal the previously frozen candidate. Source hashes and time partitions agree.
Accounting identities hold, support claims come from past fitting data, and
tests cover boundary failures. Deliver all results including negative/null
findings, a scientific figure, and a round-ten acceptance report. A newly
observed benefit on reused games is exploratory and does not change the
primary frozen round-nine conclusion.
