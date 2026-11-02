# FootballMind g-training bridge adapter

This adapter consumes the phase-4 bridge as a read-only synthetic teacher
prior.  It is a ranking aid for candidate generation when the empirical B
panel is sparse.  It is not an empirical g model and it does not change the
existing UNKNOWN support gate.

## Inputs

`SyntheticGTeacherPriorAdapter.from_path(path)` accepts either:

* `synthetic-g-model.json` with schema
  `footballmind-g-synthetic-teacher-prior-v1`; or
* `g-training-bridge.json` with schema
  `footballmind-g-training-bridge-v1`.  In that container only
  `synthetic_teacher_prior.model` is loaded.  The `empirical_refit` and
  `one_match_probe` sections remain separate evidence and are never used as
  synthetic coefficients.

The model must retain `label_origin=synthetic_domain_teacher`,
`calibration_status=not_established`, `production_ready=false`, and
`usage_rule=prior_only_until_real_independent_gains_calibrate_it`.  The
adapter records the source path, container section, and SHA-256 in its
manifest. `source_sha256` is the raw file digest; `model_sha256` is the
canonical digest of the extracted model section, so a bridge container and
its standalone model can be compared without confusing their envelopes.

Each action that opts into the prior may carry a
`synthetic_prior_features` object (the alias `g_prior_features` is accepted):

```json
{
  "information_kind": "lineup_injury",
  "uncertainty": 0.864,
  "decision_sensitivity": 0.9,
  "source_reliability": 0.82,
  "specificity": 0.88,
  "redundancy": 0.1,
  "misdirection_risk": 0.2,
  "query_cost": 0.024
}
```

The vector is explicit.  The adapter does not infer a kind or feature values
from an action ID, title, source, or the document returned later.  Missing or
invalid features produce an `UNAVAILABLE` prior record and preserve the
empirical value untouched.

## Candidate-generator boundary

The discovery generator remains a capability generator. It must not infer a
semantic kind or a numeric feature from a URL, title, anchor, or returned
article. The versioned boundary is
`football_agent.g_prior_features.attach_prior_features(actions,
features_by_action)`. It requires an exact action-id-to-vector mapping,
validates all eight fields, deep-copies the menu, and records the vector under
`synthetic_prior_features`. `EvidenceDocumentCandidates.actions_with_prior_features`
is a convenience wrapper around the version dispatcher; it does not change the
generator configuration or its discovery order. The bridge adapter documented
here consumes `footballmind-g-prior-features-v1`. The separately versioned
`footballmind-g-teaching-features-v1` contract is consumed by
`SyntheticGTeachingPriorAdapter` and has its own artifact and provenance. A
single menu may use one complete contract, but mixed schemas are rejected.

The two prior fields are decision-local metadata. The transport executor and
runtime binding strip only these declared fields when checking the immutable
request identity; tool, URL/query, information key, and cost remain bound. A
malformed vector, an unmapped action, or a duplicate alias is rejected before
the query. This keeps a missing declaration observable rather than silently
assigning a default expert opinion.

## Output contract

`adapter.assess(candidate, empirical_value=value)` returns an
`footballmind-g-prior-adapter-v1` envelope with four separate parts:

* `synthetic_prior`: the diagnostic `prior_score` and `prior_net_value`,
  `support_status=SYNTHETIC_PRIOR`, model schema/SHA-256, and explicit
  `evidence_origin=synthetic_domain_teacher`;
* `empirical`: the supplied empirical support status and values, or nulls;
* `effective`: the lane used by the existing acquisition gate; it is
  `support_status=UNKNOWN`, `expected_signed_gain=null`, and `net_voi=null`
  until the empirical estimate has the exact status `supported` and finite
  gain/net values;
* `provenance`: separate `synthetic` and `empirical` origins.

`selection_allowed` is true only for that last empirical condition.  A prior
score can therefore order a diagnostic menu with
`rank_diagnostic`, but it cannot authorize an acquisition, turn UNKNOWN into
COMPUTED, replace the constant empirical g baseline, or provide a propensity,
probability, monetary EVSI, or realized gain.

When a `ContextVOIPolicy` is constructed with the optional
`synthetic_prior=adapter`, the policy attaches this envelope under
`candidate.value.synthetic_prior`.  The policy's top-level
`expected_signed_gain`, `net_voi`, `status`, and selection decision continue
to come from the empirical model.  The action field is admitted by the
existing runtime action schema, so the feature vector is present in the
pre-query menu and is not reconstructed from the response.

## Current bridge and expected diagnostic value

The executable pre-query path is also captured in
`var/submission/p9-g-training-bridge-20261010/g-prequery-builder-example.json`.
On a frozen three-outcome public state (`[0.52, 0.23, 0.25]`) and an explicit
four-action capability registry, the teaching prior orders the diagnostic
menu `lineup > motivation > injury > weather`.  Every row still has
`selection_allowed=false` because empirical support is UNKNOWN.  This is a
mechanism trace showing where the autonomous choice is made; it is not a
claim that the synthetic score is a calibrated match-specific probability.

The adapter was checked against:

* `var/submission/p9-g-training-bridge-20261010/g-training-bridge.json`
  (SHA-256
  `e87b523b1f87e7b47a0a0578d4555d45a3a173f4eb04d19e37b926b21286c8ae`);
* `synthetic-g-model.json` (raw file SHA-256
  `68798649ee040d2ace82545121ba82e978649056cfe3aa5e9cff9a093f556dc5`,
  canonical model SHA-256
  `5b41958d623bcf9a76b0225c9fb1581068bf77bdfd6da358bcaef7cba634d95c`).

For the declared high-uncertainty lineup/injury vector above, the adapter
reports a synthetic prior score of approximately `0.0875774320` and a prior
net value of approximately `0.0635774320` after the declared experimental
cost.  These numbers are teacher-prior diagnostics only; they do not assert
that the real fixture has that expected gain.

## Next integration steps

1. Use the frozen `footballmind-g-prequery-feature-builder-v1` boundary to
   emit `synthetic_prior_features` from the public prediction and an explicit
   source-capability registry. `build_prequery_prior_features` derives only
   public-state geometry (normalized entropy, top-two probability margin, and,
   for the teaching schema, cutoff-to-kickoff hours). The registry must supply
   the information family, source reliability/specificity, redundancy,
   misdirection risk, and query cost (or the teaching model's corresponding
   action/scenario fields). It never reads an action URL, title, query text,
   or returned body for semantic inference. `EvidenceDocumentCandidates`
   exposes `actions_with_prequery_features` for exact action-id binding.
2. Keep empirical g fitting and the `support_status=UNKNOWN` gate unchanged;
   use `rank_diagnostic` only for analysis or a separately declared prior
   ordering, never as the runtime selection rule.
3. Add real independent B fixtures and timestamped before/after f gains.  A
   future calibration adapter may promote a prior only after an explicit
   calibration report and a new model artifact; it must not mutate this
   synthetic artifact.
4. Wire the same optional adapter into the historical/official policy
   constructors only when the controller freezes that interface.  Until then
   those policies remain unchanged and the current bridge is diagnostic-only.

