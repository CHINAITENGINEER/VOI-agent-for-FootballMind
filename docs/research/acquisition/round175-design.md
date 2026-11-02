# Round175: coherent evidence-strength adaptation

Design frozen before this round's fitting and evaluation, 2026-10-06.

The scientific target remains sequential information acquisition. Round174
established an executable loop but did not establish a predictive benefit.
This round tests a probability-model correction, not another cost multiplier.
Round10's fixed alpha was a loss/cost conversion factor; it did not change the
joint response likelihood or the posterior probability update.

For each already frozen joint model, let m(z|H) = sum_y p0(y|H) q(z|y,H).
Use q_s(z|y,H) = s q(z|y,H) + (1-s) m(z|H), for 0 <= s <= 1. Construct
this mixture once at the initial fixture context, then condition the same
joint model after every observation. Both correlated tool returns are retained.
The free prior and joint marginal distribution of returns are unchanged.
Posterior probabilities become s p_original(y|observations,H) + (1-s) p0(y|H).
This is an ordinary shrinkage construction, not a novel-theory claim.

Select one strength per frozen model by minimizing mean held-out log loss
over three equally weighted acquired sets: results, xG, results+xG. Every
match receives equal weight, and three states are not three independent
matches. Use the original chronological validation IDs, outside both base
and joint training. Labels must be available at the adaptation cutoff and
strictly before the evaluated prediction times. Minimize the convex one-
dimensional objective on [0,1] by endpoint checks and deterministic bisection;
choose zero for a completely flat objective. Do not choose strength or any
threshold from test outcomes, query counts, or an appealing example.

Keep the seven frozen reference models, dataset, 24h prediction cutoffs,
costs, budgets, empirical support minima and original per-fold cost weights.
The reused 290 test games are exploratory diagnostics only, even though the
parameter selection within each fold respects chronology. Compare original
strict VoI, adapted strict VoI, base-only, and all-available with both original
and adapted predictors. Preserve adverse and null results and all cases where
zero acquisition is selected. No action frequency target is imposed.

Report loss, Brier score, query counts, actual paths, net utility under the
existing reporting cost weight 0.5, and paired UTC-week bootstrap summaries
(2,000 resamples, seed 1729). Intervals are descriptive: no correction for
repeated research, parameter fitting or multiple comparisons is claimed.
All actor trajectories must finish before their outcome sidecar is scored.
Unknown open-world candidates stay unknown, and the menu stays incomplete.

For a genuinely new temporal check, freeze the last fold's reference model,
selected strength and policy configuration before discovering new outcomes.
Attempt to obtain the official next-season EPL data using existing public
source discovery and archival mechanisms. Admit a new evaluation bundle only
after verifying source season, identities, feature lineage, availability,
label dates, and disjoint match IDs. All eligible completed games before the
captured observation time enter, without selecting favorable cases. If no
valid new season can be obtained, record that explicitly and deliver the
frozen evaluation entry point; never relabel old games as a new holdout.

Acceptance: artifact round-trip and tamper detection; unchanged initial
priors and response marginals; coherent branch means at both acquisition
orders; s=0 gives no probability update or positive EVSI; s=1 reproduces the
reference; temporal and training-ID overlap rejection; no hidden outcomes at
decision time; actual-return posterior verification; reproducible reports.
Empirical support and shrinkage do not certify calibration or an open-world
optimal stopping rule. Real news/lineup/tactical response modeling and
independently corroborated outcomes remain separate research gaps.
