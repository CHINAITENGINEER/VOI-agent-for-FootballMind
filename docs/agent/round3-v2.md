# Formal active-information-v2 bridge

This bridge consumes the research release under
`var/parallel/baselines/round3/active-information-v2/` without retraining or
selecting fixtures by their outcomes. The service configuration is
`var/parallel/agent/round3/real-v2/backend-fold-2.json`.

The selected fixture is the first fold-2 scheduled fixture from the existing
v1 acceptance (`654a90d2-65a9-5cae-8494-f2fb685a122b`, cutoff
`2025-12-29T19:30:00Z`). It stays in the a3b3 formal canonical-ID namespace.
The v2 fold-2 validation report freezes `lambda_money=0.25` for the
`money=0.05`, `requests=2` budget. Platform costs are simulated reveal credits:
`results_view` costs 0.02 and reveals Elo plus recent form; `xg_view` costs
0.03 and reveals rolling xG. No HTTP request is made by this historical run.

The frozen predictor is
`partial-subset-logistic-1.0:fb968fc75bd79f62de8d8ef5abe8c57a37aae93c15222c5b08d059110e33721a`,
trained through `2025-11-29T12:00:00Z`; the gain estimator is
`empirical-mask-action-shrinkage-v1:3f28ebabe006605999da`, with the same
training cutoff and dataset
`platform-history-research-v1.0:a3065657f0a6e7e8385e783a9169e23d35e419469c54061e1da91ee4806493ad`.

The recorded executions are in `real-v2/fixed-run.json`,
`real-v2/all_candidates-run.json`, and the compact value trace is
`real-v2/v2-validation.json`:

- fixed order selected `results_view` then `xg_view`, spent 0.05 simulated
  credits, and stopped with no legal actions;
- numerical `select_or_stop` selected `xg_view` directly (`gain`
  `0.0083338731`, cost in loss units `0.0075`, net gain `0.0008338731`), then
  stopped after the revealed state made all remaining net gains non-positive.

Both final probabilities were recomputed by the bridge's `PartialPredictor`
from the final visible state and compared with a direct prediction call. The
run audit chain and evidence references remain in each run's RunStore record.
The bridge rejects incomplete or unknown cost vectors before value selection,
and emits the server's `recommended_budget` and frozen artifact hashes for UI
consumers.
