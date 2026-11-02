# Round 6: competition-context evidence projection

This change adds a small optional evidence contract. It does not change existing
`VisibleState`, acquisition catalogs, collectors, database tables, or historical
exports. Existing EPL exports contain no verified group-stage qualification,
lineup, rotation, or tactical-intention fields. The new tests are constructed
contract cases, not observations of Japan–Thailand or China–UAE, and not evidence
of improved predictive performance.

## Public API

```python
from football_platform.information.competition_context import project_competition_context

context = project_competition_context(
    fixture={
        "match_id": "example-match", "competition_id": "example-group",
        "home_team_id": "a", "away_team_id": "b",
        "kickoff": "2026-09-20T12:00:00Z",
    },
    prediction_cutoff="2026-09-20T11:00:00Z",
    observations=[{
        "evidence_id": "example-qualification", "match_id": "example-match",
        "competition_id": "example-group", "team_id": "a",
        "topic": "qualification", "value": "qualified",
        "available_at": "2026-09-20T09:00:00Z",
        "source_id": "constructed-scenario", "source_kind": "synthetic",
        "epistemic_status": "hypothesis",
    }],
    acquired_evidence_ids=["example-qualification"],
)
payload = context.model_dump(mode="json")
```

`CompetitionContext` contains a match-level `stage` claim and fixed slots under
`teams[team_id]`: `qualification`, `rank_objective`, `rotation`, `lineup`, and
`tactical_intent`. Every missing slot is `{"status": "unknown", "value": null,
...}`. `evidence` contains the accepted records keyed by inert evidence IDs.
`readiness` is `no_context_observed`, `synthetic_only`, or `partially_observed`.
The last status reports presence, not evidence sufficiency or research readiness.

Rank and rotation have independent dimensions. Use
`rank_objectives[team_id]["qualification" | "first_place"]` for the qualification
and first-place conditions, and `rotation_states[team_id]["planned" | "actual"]`
for rotation plans and observed rotation. Each dimension remains a normal claim
with its own status, value, and evidence IDs. The older slots under `teams` are
summaries: when several dimensions are present, their scalar `value` is null and
`dimension_values` contains the separate values. Such a summary may be verified
without containing a single scalar conclusion. Cross-dimension differences are
never contradictory alternatives; only disagreement within a dimension produces
`conflicting`.

For example, `draw_sufficient_for_qualification` and
`conditional_on_other_fixture` with `conditional_target="first_place"` may both
hold for the same team. Similarly, a sourced `planned` rotation claim and an
`observed` (or `not_observed`) rotation claim occupy separate phases. A plan that
was not realized does not by itself make either report false.

## Meaning of each topic

| Topic | Supported values | What this establishes |
| --- | --- | --- |
| `stage` | `group`, `league`, `knockout`, `qualifying`, `playoff`, `friendly` | Competition stage; requires `team_id=null`. |
| `qualification` | `qualified`, `not_yet_qualified`, `eliminated` | Qualification status, separately from group rank. |
| `rank_objective` | `draw_sufficient_for_qualification`, `draw_sufficient_for_first_place`, `win_required_for_qualification`, `win_required_for_first_place`, `position_locked`, `conditional_on_other_fixture` | The stated rank condition, not the team's actual utility function or tactic. |
| `rotation` | `planned`, `not_planned`, `observed`, `not_observed` | A sourced rotation claim; planned changes are distinct from observed changes. |
| `lineup` | `starting_xi` | The proposed or confirmed player set in `player_ids`, distinguished by epistemic status; no automatic squad-quality score. |
| `tactical_intent` | `attack_oriented`, `balanced`, `defense_oriented` | A pre-match hypothesis, never a verified description of full-match behavior. |

A conditional rank objective must contain `conditional_target` (`qualification`
or `first_place`), `condition_fixture_ids` for the other fixture(s), and
`premise_evidence_ids`. These fields preserve the dependency; they do not resolve
another match's future result. `objective_target` is inferred from the explicit
qualification/first-place values or the `conditional_target`; `position_locked`
requires an explicit `objective_target`. Conflicting explicit target declarations
are rejected. The module does not calculate standings or assume
any tie-break rules. Without an upstream verified source or a separately verified
complete standings/rule derivation, the caller must leave the corresponding slot
unknown or label it an unverified claim/hypothesis.

## Evidence and as-of checks

- Each accepted record has a source ID, source kind, fixture/competition/team
  association, evidence ID, and timezone-aware `available_at`. Optional publication
  and observation times must not exceed availability or the prediction cutoff.
- `available_at <= prediction_cutoff < kickoff` is enforced. Purchased future or
  post-match observations, a different match/competition, and a team outside the
  fixture raise a validation error.
- Records absent from the acquisition ledger or marked unavailable/stale/failed
  are discarded before payload validation. Malformed hidden payloads have no
  effect on the returned state.
- Hypothesis premises must be acquired and present in the accepted projection.
  Dependencies are resolved as a directed acyclic graph; missing premises and
  cyclic claims never validate one another. A verified conclusion requires verified
  premises whose claim slots are also free of conflicts; selecting one source from
  contradictory verified premises does not certify a derived conclusion.
- `verified_fact` requires an official/structured-provider record and an explicit
  upstream `verified_by` identifier. LLM, analyst, reporter, and synthetic claims
  cannot self-promote into verified facts. The owner still has to perform and
  audit the named verification; this schema does not authenticate a source.
- Probabilities belong to hypotheses, lie in `[0, 1]`, are finite, and require a
  `probability_basis`. A valid number or recorded basis is not a calibration
  guarantee. Model probabilities do not become fact-verification confidence.
- Distinct values for the same slot result in `status=conflicting`, `value=null`,
  retained evidence IDs, and explicit alternatives. The module does not silently
  choose a source or average contradictory claims. Different lineup player sets
  also conflict; simple changes in player ordering do not. Lineup confirmation is
  separate from an observed rotation claim, so those two claims are compatible.
- Evidence, source, premise, verifier, fixture/team/player, hash, and probability
  basis identifiers are bounded inert tokens; file paths, URLs, and free prose do
  not pass through the public reference fields. Evidence content remains the
  schema's allow-listed structured claims, with no raw report/label payload field.

A `source_published_at` read retrospectively from a web article is not proof that
the system actually possessed the article at the old cutoff. The caller must
retain the true operational acquisition time; historical reconstruction requires
an explicitly documented availability assumption in its experiment manifest.
This change supplies no historical availability proof for the user's examples.

## Interpretation and next integration

Qualification is potentially relevant to the *next query*: actual lineup,
rotation plan, and retained squad quality may still matter. It never directly
sets a negative strength adjustment here. Likewise, an incentive to draw does
not establish that a coach will defend throughout a match. These are distinct
claims with independent evidence and uncertainty.

The projection is suitable as an optional public context for an acquisition
policy. Feeding it into a predictive/value model remains a separately versioned
integration and evaluation step. In particular, a deterministic rule such as
`qualified -> query lineup` would not demonstrate learned conditional value of
information. Neither a single win nor a single loss establishes whether the
pre-match information had positive expected decision value.

## Verification

```text
python -m pytest tests/platform/information/test_competition_context.py --import-mode=importlib -q
60 passed
```

The tests cover independent qualification/rank/rotation/tactical claims, conditional
other-fixture rank dependencies and simultaneous qualification/first-place goals,
separate planned/actual rotation, hidden/unavailable malformed payloads, missing
or cyclic hypothesis premises, future lineup rejection, fixture/team association,
source-verification boundaries, conflicts, and explicit synthetic readiness.
No external HTTP calls, database changes, or production archive writes occur.
