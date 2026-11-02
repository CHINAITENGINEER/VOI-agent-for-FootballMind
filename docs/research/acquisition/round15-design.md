# Round 15 frozen design: conditional-information mechanism benchmark

Date: 2026-09-27. Previous acceptance SHA256: 00ed1dc2c8959e9e48eba2e7a8c3f75aaad05021e1b09804bcf7b4f0b39a294c.

Question: does the existing numerical Agent buy a weak first observation when it enables a useful second query, choose the second tool from the actual first return, and stop on redundant or irrelevant information? Separate this mechanism question from whether the learned football return/reward model estimates real value correctly.

Scope is an enumerated synthetic finite population with known conditional distributions, no new football matches, fitting, scraping, LLM calls or parameter search. Reuse the frozen `JointRiskPlanner` and durable `AcquisitionWorkflow`; add model/environment adapters and independent verification. Old source/results remain unchanged. There is no new algorithm-novelty or real-sports-effectiveness claim.

## Fixed worlds, costs and constraints

Outcome Y is a three-class variable indexed 0,1,2. Log loss uses natural logarithms. All simulated money has loss conversion lambda=1, requests cost 1, other units 0. Maximum actions=2, money caps as listed. Query returns are one-shot categorical observations. Positive entries are enumerated with their exact generating mass; zero-mass worlds are excluded. No sampling or bootstrap confidence interval is appropriate for these fully enumerated expectations.

Let K(z|y)=0.9 if z=y and 0.05 otherwise. U3 is uniform on 0,1,2.

1. `gated_stop`: Y~U3, G~Bernoulli(0.5) independent. `context` returns G and costs .02. `signal_a` costs .20; its return has K(.|Y) if G=0, otherwise U3 independent of Y. Money cap .22. This tests context-dependent continue/STOP and paying for context with zero immediate outcome information.
2. `gated_route`: same Y,G. `context` costs .02, `signal_a` and `signal_b` each cost .20. Given Y,G, signals are independent. If G=0 signal_a uses K and signal_b is U3; if G=1 the roles reverse. Money cap .42. This tests actual content selecting a different next tool; all three queries exceed the two-request cap.
3. `complementary`: `signal_a` and `signal_b` are independent U3, each costs .12. Y has K(.|(a+b) mod 3). Money cap .24. Each individual return is independent of Y; together they are informative. This separates planning depth from a content-dependent routing advantage.
4. `redundant`: Y~U3, A~K(.|Y); `signal_a` returns A at cost .04; `signal_b` returns exactly the same A at cost .06. Money cap .10. Second acquisition has zero conditional information, even though its marginal information is positive.
5. `irrelevant`: P(Y)=(.7,.2,.1); `signal_a` is independent with P(0)=.6, P(1)=.4, cost .02; `signal_b` is independent with P(0)=.3, P(1)=.7, cost .03. Signals and Y mutually independent. Money cap .05. Accurate tool observations need not change the outcome distribution.

The specification determines every row before evaluation. Do not change factors or costs to improve results. Football analogies (qualification, lineups, replacement strength/tactics) are conceptual motivation, not labels asserting these synthetic factors describe real teams.

## Models and policies

For every scenario: `correct` equals the generating distribution; `independent` is P(Y) times the product of each observation's marginal, removing informative dependence while preserving marginals. For gated scenarios also `swapped_context`: swap G=0 and G=1 in the generating distribution, reversing which context makes a tool informative. Wrong-model runs use the declared wrong distribution for BOTH forecasting and query planning. They illustrate model dependence, not an identified cause of the earlier football residuals. All true positive-mass observation patterns must retain positive model support; no outcome has zero predicted probability on a scored true world.

Evaluate five policies for all declared scenario/model pairs:

- `base-only`: no query.
- `myopic`: existing JointRiskPlanner horizon=1, reevaluated after any actual return within the overall resource cap.
- `adaptive-two-step`: existing JointRiskPlanner horizon=2, reevaluated after each actual return within the same cap.
- `best-fixed-subset`: choose the best affordable subset of at most two queries under the declared model before any returns, then execute it entirely. Final forecasts still use the actual returns.
- `best-fixed-order-stop`: select the best fixed ordered list of at most two queries under the model; allow content-dependent STOP along that list but not a change of next tool. This is a stronger control isolating adaptive tool identity from adaptive stopping. Select the list globally before observing actual returns and retain it for the episode.

Use STOP for nonpositive gains; existing planner rounds net gain to 12 decimals and breaks positive ties by action ID. For fixed baselines choose minimum rounded expected risk, then shorter list, then lexicographic list. Report ties/tolerances instead of requiring one arbitrary oracle action when several are equivalent. All declared configurations are reported; no validation selection or favourable-case filtering.

## Shared model adapter contract

`football_research/acquisition/finite_information.py` supplies:

- `build_scenarios() -> dict[str, dict]`: scenario dict has `schema='finite-information-scenario-v1'`, `scenario_id`, `actions` (JointRiskPlanner catalog), `budget_money`, `request_budget=2`, `lambda_money=1.0`, `worlds` and `models`.
- `worlds`: `{world_id, mass, outcome: int, observations: {action_id: str}}`. Only the environment/scorer accesses an instantiated world's complete returns/outcome. The planner receives a distribution over all worlds, never its current world index or hidden outcome.
- `models`: name to a plain spec `{schema='finite-information-model-v1', action_ids: list[str], rows: [{mass, outcome, observations}], ...}`. `FiniteJointModel(spec)` exposes `model_version`, `metadata`, `predict_partial`, `predict_hypothetical`, `observed_actions`, `enumerate_action`, and save/load as appropriate. Validate identities, finite masses, normalization, observation keys and categories.
- Actual model state is `{observations: {action_id: str}, action_history: list}`; it contains only acquired returns. Reject hidden/outcome/unknown state keys. Planning adds the existing `joint_return_hypothesis` marker with `hypothetical: true` and hypothetical observations; runtime prediction rejects it. Enumerations conform exactly to JointRiskPlanner's current/posterior/mass/action contracts.
- Each returned plan retains projected branches marked hypothetical; actual reveals are separately recorded and charged. Missing/unsupported observations fail explicitly, not through fabricated posterior defaults.

## Evaluation and independent checks

Save all scenario/model distributions, configuration, model/source hashes, full per-world actual-policy trajectories, weighted per-policy metrics and selected fixed orders. The score is true-population expected terminal log loss + actual simulated fee; also show declared-model predicted score, costs, query counts, first/second action distributions, and regret against the true-distribution two-query optimum.

Root independently enumerates finite policy trees directly from saved rows (not by calling the production model/planner), verifies optimal values and all observed policies, and recomputes exact true-population expectations. Verify total row mass, posterior normalization/martingale, one-shot budget feasibility, sums of actual costs, and per-path log-loss improvement. No unknown branch is assigned an observed outcome from another branch.

Agent integration uses the existing durable executor with a new finite benchmark backend. For the `correct` model and adaptive policy, cover every distinct positive-mass observation tuple once: gated_stop 6, gated_route 18, complementary 9, redundant 3, irrelevant 4, total 40 actual synthetic workflows. These are synthetic return patterns, not independent matches. Choice of representative world is by sorted observation tuple, not outcome or performance. Full hidden tuple lives in environment only; outcome is never supplied to backend prediction, estimation or acquisition.

Agent output records candidate scores, simulated branches, actual receipts, actual updated posterior, next query/STOP and audit references. Root checks all these against the independent population oracle and research paths. Test wrong/malformed input, actual/hypothetical boundary, unsupported return, cost accounting, duplicate query, max actions and persistence. Mechanism examples should exhibit differing next actions for the same first query under different returns; redundant/irrelevant cases should preserve STOP where optimal.

## Ownership and acceptance

Research agent owns new finite model/scenarios, experiment_round15, associated research tests and method note. Agent worker owns new finite backend/workflow adapter, verification runner, Agent tests and method note. Root owns this frozen design, independent finite-tree reference, integrated analysis/figure, root verification, final report/status and acceptance. All work is limited to new files and the two cumulative status notes after prior hashes pass; frozen round14 files are read-only.

Acceptance requires numerical oracle agreement and exact weighted metrics, complete 40-pattern Agent execution with only acquired evidence, appropriate tests passing, old hash preservation, and explicit distinction between a controlled mechanism result and unestablished real-football benefit. Report any failure directly; do not adjust the predeclared scenario suite in response to outcomes.
