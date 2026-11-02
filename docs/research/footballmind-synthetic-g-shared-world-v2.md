# FootballMind synthetic g v2 research artifact

`synthetic-g-shared-world-v2` is a mechanism-teaching and evaluation artifact
for the information-value layer. It is not a football model, an empirical g,
or an online selection policy.

The simulator gives every candidate action in a state the same latent
16-world distribution and the same outcome law. Actions differ only in their
declared observation channel, reliability, response probability, update trust,
freshness and cost. The environment reliability can differ from the updater's
trust, so a source can have negative expected signed log-loss gain. A failed
request leaves the predictor unchanged. Decisions rank predicted gross gain
minus explicit cost and include `STOP`.

Training is grouped by `physical_match_id`: 60% train, 20% validation and 20%
test. Tree size and the stop margin are chosen on validation; the test partition
is evaluated after those values are frozen. The constant and STOP baselines use
the same frozen stop margin as the learned policy.

The artifact is diagnostic only. Its latent probabilities, effect vectors and
reliability fields are constructed mechanism inputs. A live candidate must
provide equivalent fields explicitly before query time or mark them unavailable;
the model must never infer them from a title, URL, query, body or response.
The empirical g UNKNOWN/constant gate and the existing v2 pre-query contract
are unchanged.

See the generated report and manifest in
`var/submission/p9-g-shared-world-v2-20261010/`.
