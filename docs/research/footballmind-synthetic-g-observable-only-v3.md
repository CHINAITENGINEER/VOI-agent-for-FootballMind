# FootballMind synthetic g observable-only v3

## Purpose

This experiment trains a conditional information-value estimator while
removing the shared-world teacher's hidden mechanism variables from the
learner input. The learner sees only the pre-query declarations that an
online caller can provide explicitly:

- current public 1X2 probabilities and derived normalized entropy;
- source reliability, freshness, redundancy, response probability and
  updater trust;
- time to kickoff;
- a one-hot action declaration for lineup, injury, motivation or weather.

`latent_probability` and `effect_logits` remain teacher-only fields. They are
not copied into the `footballmind-g-observable-features-v3` vector and cannot
be reconstructed from an action id, URL, title, query, response or document.

## Target and evaluation

The constructed shared-world teacher supplies

`E[log f_after(Y) - log f_before(Y)]`

for each candidate before request cost. The policy ranks predicted gross gain
minus the explicit query cost and may choose `STOP`. Matches, rather than
individual action rows, are split into train, validation and test groups.
The validation set selects tree leaf size and the STOP margin; the test set is
read only after those choices are frozen. Portable JSON-tree inference is
checked against the in-memory learner.

The report includes both overall harmful-query rate and the more useful
`harmful_query_given_query`, as well as positive-net precision, oracle-query
recall, false-query-given-oracle-stop, regret, calibration error and a grouped
bootstrap interval. A high conditional harmful-query rate is a failure signal,
not evidence of a safe policy.

## Provenance boundary

The teacher and all labels are synthetic. Observable declarations are also
constructed and are not verified football facts. The resulting artifact is a
research diagnostic with `production_ready=false`,
`empirical_selection_allowed=false`, and `support_status=
synthetic_research_only`. It does not alter the empirical g UNKNOWN/constant
gate, the strict pre-query v2 contract, or formal runtime.

The first 600-match run is deliberately retained as an honest ablation. Its
observable-only learner improves over a STOP/constant policy in mean synthetic
net value, but selected-query conditional harm remains material. This means
the next research task is risk-aware multi-step evidence modelling and real
before/after labels, rather than promoting this prior into online selection.
