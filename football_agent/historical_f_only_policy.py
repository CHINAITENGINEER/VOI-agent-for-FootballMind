"""Frozen historical text f for prospective exploration; the bundled g is inert.

This is an opt-in research adapter. The market snapshot remains an independent
baseline; f keeps its training-domain A-learned base. No gain is manufactured.
"""
from copy import deepcopy

from football_research.acquisition.state import parse_time
from .context_voi_policy import ContextVOIPolicy, _digest
from .historical_context_policy import HistoricalContextVOIPolicy

VERSION = "historical-f-only-research-policy-v1"


class HistoricalFOnlyPolicy(HistoricalContextVOIPolicy):
    predictor_only = True

    def __init__(self, base_policy, model, *, fixture, artifact):
        super().__init__(base_policy, model, fixture=fixture, artifact=artifact)
        # Validate the original container, but never activate or archive it as g.
        self.value_model = None
        self.value_bytes = None
        self.manifest.update(schema=VERSION, value_model_version=None,
            value_scope="unknown_not_estimated", selection_authority="controlled_exploration_only",
            predictor_only=True, bundled_g_enabled=False,
            probability_basis="historical_text_f_with_A_learned_base",
            empirical_training_eligibility="requires_separate_verified_outcome_and_trajectory_audit")
        self.manifest.pop("predictor_value_gate", None)
        self.version = VERSION + ":" + _digest(self.manifest)[:24]

    def check_start(self, at):
        at = parse_time(at)
        meta = self.predictor.metadata
        if not (parse_time(meta["f_training_as_of"]) <= parse_time(meta["fit_execution_at"]) <= at
                and parse_time(self.model.frozen_at) <= at <= parse_time(self.model.cutoff)
                and at < parse_time(self.fixture["kickoff"])):
            raise ValueError("Frozen f and market initialization must precede the valid acquisition start")

    def predict_public(self, public):
        result = super().predict_public(public)
        result.pop("gain_model_summary", None)
        result.update(predictor_only=True, gain_model_enabled=False,
            market_baseline_probabilities=list(self.model.probabilities))
        return result

    # The existing no-value-model branch preserves UNKNOWN and budget checks.
    plan_public = ContextVOIPolicy.plan_public

    def journal_initial_assessment(self):
        return {"status": "PREDICTOR_ONLY",
            "prediction": dict(zip(("home", "draw", "away"), self.predictor.metadata["empirical_base"])),
            "candidate_expected_gains": None, "model_versions": [self.predictor.model_version]}


def fixture_from_verified_identity(base_policy, binding):
    """Project the existing verified ordered identity, never invent a match ID."""
    if not isinstance(binding, dict) or binding.get("status") != "CONFIRMED":
        raise ValueError("f-only acquisition requires a confirmed independent fixture identity")
    if binding.get("initialization_manifest") != base_policy.manifest:
        raise ValueError("f-only fixture identity must bind the exact market initialization")
    canonical = binding["canonical"]
    mid = canonical.get("source_event_id")
    if not isinstance(mid, str) or not mid:
        raise ValueError("A verified physical event identity is required")
    market = base_policy.model.fixture
    if parse_time(canonical["kickoff"]) != parse_time(market["kickoff"]):
        raise ValueError("f-only kickoff differs from the verified fixture")
    return {"match_id": market["match_id"], "physical_match_id": mid,
        "home_team_id": market["home_team_id"], "away_team_id": market["away_team_id"],
        "competition_id": market["competition_id"], "kickoff": market["kickoff"],
        "home_name": canonical["home_name"], "away_name": canonical["away_name"],
        "home_aliases": deepcopy(binding["aliases"]["home"]),
        "away_aliases": deepcopy(binding["aliases"]["away"]),
        "competition": canonical.get("competition"), "season": canonical.get("season")}
