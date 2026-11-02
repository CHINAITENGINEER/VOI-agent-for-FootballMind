"""Prospective market initialisation and bounded collection using existing runtime.

No text value is invented here. A frozen observed regulation 1X2 quote supplies
the initial forecast; the existing sampler acquires currently unmodelled text.
Settlement remains an independent future supervisor input. Unknown fixture
score semantics are retained, never promoted into regulation outcome labels.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.joint_planner import UNITS, _number
from football_research.acquisition.state import GROUP_FIELDS, parse_time, sanitize_state
from football_research.acquisition.value import _canonical
from football_research.data_adapters.types import ResearchDataset, json_record

VERSION = "prospective-market-prior-policy-v1"
SETTLEMENT_RULE = "90_minutes_plus_stoppage_excludes_extra_time_penalties"


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _save(path, value):
    from football_spider.storage.raw_archive import atomic_write
    atomic_write(Path(path), (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      indent=2, allow_nan=False) + "\n").encode())


def _initial_state(fixture, cutoff):
    return sanitize_state({"match_id": fixture["match_id"], "prediction_cutoff": cutoff,
        "information_mode": "operational", "base": {k: fixture[k] for k in
            ("competition_id", "home_team_id", "away_team_id", "kickoff")},
        "groups": {g: {"status": "not_acquired"} for g in GROUP_FIELDS}})


class _FrozenMarketForecast:
    def __init__(self, fixture, quote, *, frozen_at, prediction_cutoff):
        self.fixture, self.quote = json.loads(_canonical(fixture)), json.loads(_canonical(quote))
        fixture, quote = self.fixture, self.quote
        self.frozen_at = parse_time(frozen_at).isoformat()
        self.cutoff = parse_time(prediction_cutoff).isoformat()
        self.initial = _initial_state(fixture, self.cutoff)
        now, kickoff = parse_time(self.frozen_at), parse_time(fixture["kickoff"])
        if (fixture.get("information_mode") != "operational" or fixture.get("status") != "scheduled"
                or any(fixture.get(k) is not None for k in ("home_goals", "away_goals", "result_available_at"))
                or fixture.get("home_team_id") == fixture.get("away_team_id")
                or not parse_time(fixture["fixture_available_at"]) <= now < parse_time(self.cutoff) < kickoff
                or not fixture.get("input_manifest")):
            raise ValueError("An actually observed upcoming unlabelled operational fixture is required")
        if (quote.get("match_id") != fixture["match_id"] or quote.get("information_mode") != "operational"
                or quote.get("period") != "regulation" or quote.get("settlement_rule") != SETTLEMENT_RULE
                or not quote.get("input_manifest") or not quote.get("provider")
                or parse_time(quote["observed_at"]) > now
                or now - parse_time(quote["observed_at"]) > timedelta(hours=1)
                or quote.get("source_timestamp") is not None and parse_time(quote["source_timestamp"]) > now):
            raise ValueError("A recent observed regulation quote for exactly this fixture is required")
        prices = [quote.get(k) for k in ("home_price", "draw_price", "away_price")]
        if any(type(p) not in {int, float} or not math.isfinite(p) or p <= 1 for p in prices):
            raise ValueError("Three finite decimal odds greater than one are required")
        inverse = [1. / p for p in prices]
        self.probabilities = [p / math.fsum(inverse) for p in inverse]
        self.binding = {"schema": "observed-regulation-market-prior-v1", "fixture": self.fixture,
            "quote": self.quote, "frozen_at": self.frozen_at, "prediction_cutoff": self.cutoff,
            "normalization": "reciprocal_prices_divided_by_sum", "probabilities": self.probabilities,
            "quote_sha256": _digest(self.quote), "training_match_ids": [],
            "fitted_parameters": False, "calibration_status": "not_established"}
        self.model_version = "observed-market-prior:" + _digest(self.binding)
        self.metadata = {"training_as_of": self.frozen_at, "training_match_ids": [],
            "base_training_match_ids": [], "information_mode": "operational",
            "timestamp_meaning": "deterministic_quote_freeze_not_statistical_training"}

    def predict_partial(self, state):
        clean = sanitize_state(state)
        if any(clean[k] != self.initial[k] for k in self.initial):
            raise ValueError("Frozen market initialisation cannot change fixture or acquire structured features")
        return {"probabilities": list(self.probabilities), "model_version": self.model_version,
                "probability_basis": "observed_market_snapshot", "unknown_identity_fields": [],
                "information_mode": "operational", "calibration_status": "not_established"}

    def observation_support(self, state):
        self.predict_partial(state)
        return {"kernel_effective_matches": 0., "class_kernel_weight": {str(i): 0. for i in range(3)},
                "reason": "response_model_unavailable"}

    def observed_actions(self, state):
        self.predict_partial(state)
        return []


class MarketPriorPolicy(LearnedJointVOIPolicy):
    """Existing policy protocol with a market p0 and honest UNKNOWN text VoI."""
    VERSION = VERSION

    def __init__(self, fixture, quote, *, frozen_at, prediction_cutoff, lambda_cost=None, loss="log_loss"):
        self.model = _FrozenMarketForecast(fixture, quote, frozen_at=frozen_at,
                                           prediction_cutoff=prediction_cutoff)
        if loss not in {"log_loss", "brier"}:
            raise ValueError("Unsupported predictive loss")
        lambda_cost = {"money": 1.} if lambda_cost is None else lambda_cost
        if not isinstance(lambda_cost, Mapping) or set(lambda_cost) - set(UNITS):
            raise ValueError("Unknown cost unit")
        self.loss = loss
        self.lambda_cost = {u: _number(lambda_cost.get(u, 0.), u) for u in UNITS}
        self.min_effective_matches = self.min_class_weight = 1.
        self.manifest = {"schema": VERSION, "model_version": self.model.model_version,
            "trained_as_of": self.model.frozen_at, "market_initialization": self.model.binding,
            "loss": loss, "lambda_cost": self.lambda_cost,
            "min_effective_matches": 1., "min_class_weight": 1.,
            "response_model": None, "value_scope": "unknown_until_response_model_is_trained",
            "calibration_status": "not_established", "parameter_uncertainty": "not_estimated"}
        self.version = VERSION + ":" + _digest(self.manifest)[:20]
        self.cost_version = "declared-loss-conversion:" + _digest({"loss": loss, "weights": self.lambda_cost})[:20]

    @classmethod
    def from_manifest(cls, manifest):
        if manifest.get("schema") != VERSION:
            raise ValueError("Unsupported market prior policy")
        binding = manifest["market_initialization"]
        result = cls(binding["fixture"], binding["quote"], frozen_at=binding["frozen_at"],
                     prediction_cutoff=binding["prediction_cutoff"],
                     lambda_cost=manifest["lambda_cost"], loss=manifest["loss"])
        if result.manifest != manifest:
            raise ValueError("Market initialization manifest changed")
        return result

    def plan(self, state, actions, budget, *, candidate_generation_complete=False):
        if any(a.get("group_ids") or a.get("tool") not in {"read_url", "search_web"} for a in actions):
            raise ValueError("Market initialisation only permits unmodelled public web actions")
        result = super().plan(state, actions, budget, candidate_generation_complete=candidate_generation_complete)
        result["schema"] = VERSION
        result["selection_scope"] = "bounded_exploration_pending_learned_response_model"
        result["decision_digest"] = _digest({k: v for k, v in result.items() if k != "decision_digest"})
        return result

    def prior_binding(self):
        """Existing response-dataset provenance sidecar; no outcome labels."""
        return {"probability_basis": "observed_market_snapshot", "model_version": self.model.model_version,
            "model_sha256": _digest(self.model.binding), "training_as_of": self.model.frozen_at,
            "training_match_ids": [], "fitted_parameters": False,
            "initialization_kind": "observed_market_snapshot", "quote_sha256": _digest(self.model.quote)}


def select_initializations(dataset, *, as_of, limit=3, horizon_hours=48, cutoff_lead_minutes=1,
                           fixture_ids=None):
    """Deterministic upcoming selection from a verified operational projection.

    Fixture score_period can remain unknown: the forecast's target comes from
    the explicitly regulation quote. Future settlement must establish its own
    regulation semantics; this function does not settle or train anything.
    """
    if not isinstance(dataset, ResearchDataset) or type(limit) is not int or limit < 1:
        raise ValueError("A verified normalized dataset and positive limit are required")
    if (type(horizon_hours) not in {float, int} or not 0 < horizon_hours <= 1008
            or type(cutoff_lead_minutes) not in {float, int} or not 0 < cutoff_lead_minutes <= 60):
        raise ValueError("Bounded future horizon and positive prematch cutoff lead required")
    if fixture_ids is not None:
        if (not isinstance(fixture_ids, (list, tuple, set, frozenset)) or not fixture_ids
                or any(not isinstance(mid, str) or not mid.strip() for mid in fixture_ids)):
            raise ValueError("fixture_ids must be a nonempty collection of exact match IDs")
        fixture_ids = sorted(set(fixture_ids))
    allowed_fixture_ids = None if fixture_ids is None else set(fixture_ids)
    now = parse_time(as_of)
    quotes = {}
    for quote in dataset.odds:
        if quote.observed_at <= now and (quote.match_id not in quotes or
                                        quote.observed_at > quotes[quote.match_id].observed_at):
            quotes[quote.match_id] = quote
    selected, excluded = [], []
    matches = sorted(dataset.matches, key=lambda m: (m.kickoff, m.match_id))
    for match in matches:
        if allowed_fixture_ids is not None and match.match_id not in allowed_fixture_ids:
            continue
        if match.status != "scheduled" or not now + timedelta(minutes=30) < match.kickoff <= now + timedelta(hours=horizon_hours):
            continue
        if match.match_id not in quotes:
            excluded.append({"match_id": match.match_id, "reason": "complete_current_regulation_quote_missing"})
            continue
        try:
            policy = MarketPriorPolicy(json_record(match), json_record(quotes[match.match_id]),
                frozen_at=now, prediction_cutoff=match.kickoff-timedelta(minutes=cutoff_lead_minutes))
        except ValueError as exc:
            excluded.append({"match_id": match.match_id, "reason": str(exc)})
            continue
        if len(selected) < limit:
            selected.append(policy.manifest)
    return {"schema": "prospective-text-initializations-v1", "created_at": now.isoformat(),
        "dataset_version": dataset.dataset_version, "selection_rule": "earliest_eligible_kickoff_then_match_id",
        "selection_scope": {"fixture_ids": fixture_ids,
            "window_start": now.isoformat(), "window_end": (now+timedelta(hours=horizon_hours)).isoformat(),
            "minimum_lead_minutes": 30, "cutoff_lead_minutes": cutoff_lead_minutes,
            "missing_fixture_ids": sorted((allowed_fixture_ids or set()) - {m.match_id for m in matches}),
            "selection_inputs": ["predeclared_fixture_ids", "kickoff", "source_status", "observed_regulation_quote"],
            "outcome_used_for_selection": False},
        "horizon_hours": horizon_hours, "limit": limit, "selected": selected, "excluded": excluded,
        "labels": [], "prior_bindings": {m["model_version"]: MarketPriorPolicy.from_manifest(m).prior_binding() for m in selected},
        "independent_result_verification_required": True, "numeric_voi_training_eligible": False}


def verify_prior_archive(manifest, archive_root):
    """Reconstruct each frozen quote and fixture from its original source bytes."""
    from .spftll import load_spftll_archive
    policy = MarketPriorPolicy.from_manifest(manifest)
    data = load_spftll_archive(Path(archive_root) / "raw", as_of=parse_time(policy.model.frozen_at))
    matches = {m.match_id: json.loads(_canonical(json_record(m))) for m in data.matches}
    if matches.get(policy.model.fixture["match_id"]) != policy.model.fixture:
        raise ValueError("Frozen fixture does not reproduce verified raw source")
    if policy.model.quote not in [json.loads(_canonical(json_record(q))) for q in data.odds]:
        raise ValueError("Frozen quote does not reproduce verified raw source")
    return {"status": "verified", "raw_observations": data.audit["verified_observations"],
            "model_version": policy.model.model_version, "quote_sha256": _digest(policy.model.quote)}


def verify_run_identity(folder, config):
    """Recheck opt-in independent identity from locally copied original bytes."""
    binding = config.get("supervisor_metadata", {}).get("verified_fixture_identity")
    if binding is None:
        return None  # Legacy records remain explicitly single-provider runs.
    from .verified_fixtures import verify_identity_binding
    manifest = config["policy_manifest"]
    if config['schema'] == 'learned-voi-open-execution-v5':
        from .evidence_collection import verify_evidence_run
        return verify_evidence_run(folder, config)
    if config["schema"] in {"learned-voi-open-execution-v3", "learned-voi-open-execution-v4"}:
        manifest = manifest["base_policy"]
    if binding.get("initialization_manifest") != manifest:
        raise ValueError("Independent identity is bound to a different initialization")
    verified = verify_identity_binding(binding, archive_root=Path(folder)/"identity-archive",
                                       as_of=config["created_at"])
    if verified["status"] != "CONFIRMED":
        raise ValueError("Acquisition fixture is not independently confirmed")
    identity = config["evidence_projection"]["match"]
    for side in ("home", "away"):
        names = {manifest["market_initialization"]["fixture"][side + "_team_name"]}
        if binding.get("schema") == "premier-league-fixture-identity-v1":
            names.add(binding["canonical"][side + "_name"])
        if (identity[side + "_team"] not in names
                or identity.get(side + "_aliases") != binding["aliases"][side]):
            raise ValueError("Candidate aliases differ from the verified match identity")
    if parse_time(identity["kickoff_at"]) != parse_time(binding["canonical"]["kickoff"]):
        raise ValueError("Candidate kickoff differs from the verified match identity")
    return verified


def _copy_observations(source, target, *, observation_ids):
    found = set()
    for obs in source.observations():
        if obs.id not in observation_ids:
            continue
        target.save(obs.source, obs.endpoint, obs.params, source.read_body(obs),
            status=obs.http_status, headers=obs.response_headers, request_id=obs.request_id,
            context=obs.context, error_class=obs.error_class, observed_at=obs.observed_at,
            observation_id=obs.id)
        found.add(obs.id)
    if found != set(observation_ids):
        raise ValueError("Identity observation archive is incomplete")


def verify_run_learning(folder, config):
    """Verify an opt-in cohort from local snapshots, never from a live registry.

    The declared registry scope is preserved; this cannot discover omitted
    historical exposure. Cohort roles remain supervisor-only information.
    """
    metadata = config.get("supervisor_metadata", {})
    cohort = metadata.get("learning_cohort")
    if cohort is None:
        return None
    from .prospective_learning import validate_learning_plan
    folder = Path(folder)
    raw = (folder / "learning-context.json").read_bytes()
    if sha256(raw).hexdigest() != metadata["input_bindings"].get("learning_context_sha256"):
        raise ValueError("Learning context differs from the pre-query commitment")
    context = json.loads(raw)
    mid = config["initial_state"]["match_id"]
    identity = metadata.get("verified_fixture_identity")
    if identity is None or context["identity_bundle"]["bindings"].get(mid) != identity:
        raise ValueError("Learning cohort identity differs from the acquired fixture")
    checked = validate_learning_plan(context["learning_plan"],
        initialization_plan=context["initialization_plan"], identity_bundle=context["identity_bundle"],
        previous_plans=context["previous_plans"], as_of=config["created_at"],
        artifact_paths={key: folder / ("learning-" + key.replace("_", "-") + ".json")
                        for key in context["learning_plan"]["artifacts"]})
    if checked["by_match"].get(mid) != cohort:
        raise ValueError("Run learning cohort differs from its registered purpose")
    for key, config_key in (("text_model", "text_response_model"), ("risk_model", "text_risk_model")):
        model, archived = checked[key], config.get(config_key)
        version = None if model is None else model.model_version
        if (archived is None) != (model is None) or archived is not None and archived["model_version"] != version:
            raise ValueError("Runtime did not use the learning cohort's frozen model")
    return checked["binding"]


def collect_prepared_batch(plan_path, *, output, archive_root, allowed_hosts,
                           seed_urls=(), requests_per_match=2, seed=179, route="environment",
                           identity_bundle=None, identity_archive=None, news_provider=None,
                           source_profile_version=1, news_projection_version=1, evidence_version=None,
                           learning_plan_path=None, previous_learning_plan_paths=(),
                           candidate_discovery_version=None, context_registration_path=None,
                           context_predictor_path=None, news_route_overrides=None,
                           context_confirmation_path=None, context_value_model_path=None,
                           policy_arm="voi", policy_arm_version="footballmind-policy-arm-v1",
                           arm_assigned_at=None, random_seed=None, propensity_logging=True,
                           arm_schedule=None, journal_enabled=False,
                           observable_shadow_path=None, f_only_predictor_path=None,
                           f_only_semantic_config_path=None,
                           f_only_replay_trace_path=None,
                           training_exploration_path=None):
    """Execute saved p0 entries using real clocks, existing tools and runtime.

    Model selection is frozen before any web I/O. The caller supplies source
    capabilities; no user-generated research question or topic weights exist.
    A completed output is immutable and must be replayed, never re-executed.
    ``training_exploration_path`` is an explicit data-collection-only opt-in;
    omitting it preserves the deployment selector and all historical behavior.
    """
    from football_agent.inquiry_transport import RoutedOpenWebTools
    from football_agent.learned_voi_runtime import LearnedVOIRuntime, digest
    from football_agent.learned_voi_web import MatchDocumentCandidates, ArchivedAndWebExecutor
    from football_agent.match_analysis_report import write_analysis_report
    from football_spider.storage.raw_archive import RawArchive
    plan_path, output = Path(plan_path), Path(output)
    if output.exists():
        raise FileExistsError("Batch directory exists; resume verification instead of repeating I/O")
    from football_agent.learned_voi_runtime import POLICY_ARMS
    if policy_arm not in POLICY_ARMS:
        raise ValueError("Policy arm must be one of: " + ", ".join(POLICY_ARMS))
    if not isinstance(policy_arm_version, str) or not policy_arm_version:
        raise ValueError("A policy arm version is required")
    if type(propensity_logging) is not bool:
        raise ValueError("propensity_logging must be boolean")
    if type(journal_enabled) is not bool:
        raise ValueError("journal_enabled must be boolean")
    training_exploration_spec = None
    if training_exploration_path is not None:
        training_exploration_path = Path(training_exploration_path)
        training_exploration_spec = json.loads(training_exploration_path.read_text(encoding="utf-8"))
        if not isinstance(training_exploration_spec, dict) or \
                training_exploration_spec.get("schema") != "footballmind-g-training-exploration-config-v1":
            raise ValueError("training exploration config schema is invalid")
        if training_exploration_spec.get("enabled") is not True:
            raise ValueError("training exploration requires explicit enabled=true")
        epsilon = training_exploration_spec.get("epsilon", 0.2)
        if type(epsilon) not in {int, float} or not math.isfinite(float(epsilon)) or not 0.0 < float(epsilon) <= 1.0:
            raise ValueError("training exploration epsilon must be in (0, 1]")
        exploration_seed = training_exploration_spec.get("seed", seed)
        if type(exploration_seed) is not int or not 0 <= exploration_seed < 2**256:
            raise ValueError("training exploration seed must be an integer in [0, 2**256)")
        if not propensity_logging:
            raise ValueError("training exploration requires propensity logging")
        if not journal_enabled:
            raise ValueError("training exploration requires journal_enabled=true")
    if f_only_semantic_config_path is not None and f_only_predictor_path is None:
        raise ValueError("Entity semantic shadow configuration requires an explicit f artifact")
    if f_only_replay_trace_path is not None and f_only_semantic_config_path is None:
        raise ValueError("Entity semantic replay requires an explicit semantic configuration")
    if journal_enabled and any(value is not None for value in
            (learning_plan_path, context_registration_path, context_confirmation_path)):
            raise ValueError("Journal collection is separate from bundled learning/context registration cohorts")
    if f_only_predictor_path is not None:
        if any(value is not None for value in (learning_plan_path, context_registration_path,
                context_confirmation_path, context_predictor_path, context_value_model_path,
                observable_shadow_path)):
            raise ValueError("f-only collection is separate from existing f/g cohorts and shadow policies")
        if not journal_enabled or identity_bundle is None or evidence_version != "match-evidence-extraction-v5":
            raise ValueError("f-only collection requires journal, verified identities and body evidence v5")
    f_only_replay_traces = None
    f_only_replay_bytes = None
    if f_only_replay_trace_path is not None:
        f_only_replay_trace_path = Path(f_only_replay_trace_path)
        f_only_replay_bytes = f_only_replay_trace_path.read_bytes()
        loaded_replay = json.loads(f_only_replay_bytes)
        if not isinstance(loaded_replay, dict):
            raise ValueError("Entity semantic replay must be a JSON object")
        # A single trace is convenient for the one-match case. A multi-match
        # file uses an explicit match_id map and never guesses by ordering.
        if loaded_replay.get("schema", "").startswith("entity-semantic-shadow-extraction-trace-"):
            f_only_replay_traces = {"__single__": loaded_replay}
        else:
            f_only_replay_traces = loaded_replay.get("traces")
            if not isinstance(f_only_replay_traces, dict) or not f_only_replay_traces:
                raise ValueError("Entity semantic replay requires a trace or a nonempty traces map")
            if any(not isinstance(value, dict) for value in f_only_replay_traces.values()):
                raise ValueError("Entity semantic replay traces must be JSON objects")
    if type(requests_per_match) is not int or not 1 <= requests_per_match <= 4:
        raise ValueError("At most four public attempts per match")
    from football_agent.match_evidence import SUPPORTED_VERSIONS
    if evidence_version is not None and evidence_version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported match evidence version")
    if candidate_discovery_version is not None:
        if type(candidate_discovery_version) is not int or candidate_discovery_version not in set(range(1, 15)):
            raise ValueError("Candidate discovery version must be from 1 to 14")
        if identity_bundle is None:
            raise ValueError("Explicit evidence candidates require independently verified identities")
        if candidate_discovery_version == 5 and evidence_version != 'match-evidence-extraction-v4':
            raise ValueError("Candidate discovery version 5 requires content-role evidence version 4")
        if candidate_discovery_version == 6 and evidence_version != 'match-evidence-extraction-v5':
            raise ValueError("Candidate discovery version 6 requires HTML-body evidence version 5")
        if candidate_discovery_version in {7, 8, 9, 10, 11, 12, 13, 14} and evidence_version != 'match-evidence-extraction-v5':
            raise ValueError("Candidate discovery versions 7 to 12 require HTML-body evidence version 5")
    if type(news_projection_version) is not int or news_projection_version not in set(range(1, 14)):
        raise ValueError("News projection version must be between 1 and 13")
    if news_route_overrides is not None and news_provider is None:
        raise ValueError("Explicit news source routes require a news provider")
    if news_projection_version >= 2 and news_provider is None:
        raise ValueError("Versioned news projection requires a registered news provider")
    if type(source_profile_version) is not int or source_profile_version not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("Source profile version must be from 1 to 9")
    if source_profile_version >= 2 and identity_bundle is None:
        raise ValueError("Source profile versions 2, 3 and 4 require independently verified identities")
    context_mode = context_registration_path is not None or context_confirmation_path is not None
    if context_registration_path is not None and context_confirmation_path is not None:
        raise ValueError("Context B and C collection are mutually exclusive")
    if context_mode != (context_predictor_path is not None):
        raise ValueError("Context collection needs registration and its frozen predictor")
    if (context_confirmation_path is None) != (context_value_model_path is None):
        raise ValueError("Context confirmation needs registration and its frozen value model")
    if context_mode and (learning_plan_path is not None or identity_bundle is None):
        raise ValueError("Context B/C is separate from legacy cohorts and requires verified identities")
    if previous_learning_plan_paths and learning_plan_path is None and not context_mode:
        raise ValueError("Previous learning plans require a current learning plan")
    if learning_plan_path is not None and identity_bundle is None:
        raise ValueError("Learning cohorts require independently verified identities")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != "prospective-text-initializations-v1" or not plan.get("selected"):
        raise ValueError("A nonempty precommitted initialization batch is required")
    if len(plan["selected"]) * requests_per_match > 10:
        raise ValueError("A collection batch is bounded to ten HTTP attempts")
    policies = [MarketPriorPolicy.from_manifest(m) for m in plan["selected"]]
    if len({p.model.fixture["match_id"] for p in policies}) != len(policies):
        raise ValueError("Duplicate match initialization cannot inflate sample count")
    now = datetime.now(timezone.utc)
    arm_assigned_at = arm_assigned_at or now.isoformat()
    if parse_time(arm_assigned_at) > now:
        raise ValueError("Policy arm assignment must be frozen before collection starts")
    arm_seed_base = seed if random_seed is None else random_seed
    if type(arm_seed_base) is not int or not 0 <= arm_seed_base < 2**256:
        raise ValueError("Random seed must be an integer in [0, 2**256)")
    if arm_schedule is not None:
        if not isinstance(arm_schedule, (list, tuple)) or len(arm_schedule) != len(policies):
            raise ValueError("arm_schedule must contain one predeclared assignment per selected match")
        for assignment in arm_schedule:
            if isinstance(assignment, str):
                arm_name = assignment
            elif isinstance(assignment, dict):
                arm_name = assignment.get("policy_arm")
            else:
                raise ValueError("Each arm_schedule entry must be a policy arm string or mapping")
            if arm_name not in POLICY_ARMS:
                raise ValueError("arm_schedule contains an unsupported policy arm")
    else:
        arm_schedule = [policy_arm] * len(policies)
    arm_assignments = []
    for index, assignment in enumerate(arm_schedule):
        if isinstance(assignment, str):
            assignment = {"policy_arm": assignment}
        else:
            assignment = dict(assignment)
        assignment.setdefault("policy_arm_version", policy_arm_version)
        assignment.setdefault("arm_assigned_at", arm_assigned_at)
        assignment.setdefault("random_seed", (arm_seed_base + index) % 2**256)
        assignment.setdefault("propensity_logging", propensity_logging)
        if assignment["policy_arm_version"] != policy_arm_version:
            raise ValueError("All arm assignments must use the batch policy_arm_version")
        if type(assignment["propensity_logging"]) is not bool:
            raise ValueError("arm assignment propensity_logging must be boolean")
        assignment["arm_assigned_at"] = assignment["arm_assigned_at"] or arm_assigned_at
        if parse_time(assignment["arm_assigned_at"]) > now:
            raise ValueError("Every policy arm assignment must be frozen before collection starts")
        if type(assignment["random_seed"]) is not int or not 0 <= assignment["random_seed"] < 2**256:
            raise ValueError("Every arm assignment random seed must be an integer in [0, 2**256)")
        assignment["match_index"] = index
        arm_assignments.append(assignment)
    verified = {}
    if f_only_predictor_path is not None and any(
            a["policy_arm"] not in {"random_explore", "fixed_order", "full_probe"}
            or a["propensity_logging"] is not True for a in arm_assignments):
        raise ValueError("f-only collection requires controlled exploration with propensity logging")
    if f_only_predictor_path is not None and any(
            a["policy_arm"] in {"fixed_order", "full_probe"}
            and a["policy_arm_version"] != "footballmind-policy-arm-v2" for a in arm_assignments):
        raise ValueError("f-only deterministic propensity logging requires footballmind-policy-arm-v2; "
                         "experiment labels must not replace the executable policy arm version")
    if (identity_bundle is None) != (identity_archive is None):
        raise ValueError("Independent identity needs both its bindings and raw archive")
    if identity_bundle is not None:
        from .verified_fixtures import verify_identity_binding
        bundle = json.loads(Path(identity_bundle).read_text(encoding="utf-8"))
        for policy in policies:
            binding = bundle.get("bindings", {}).get(policy.model.fixture["match_id"])
            if (not isinstance(binding, dict) or binding.get("initialization_manifest") != policy.manifest):
                raise ValueError("The entire collection batch requires matching independent identities")
            checked = verify_identity_binding(binding, archive_root=identity_archive, as_of=now)
            if checked["status"] != "CONFIRMED":
                raise ValueError("Disputed or unconfirmed fixtures cannot enter verified collection")
            verified[policy.model.fixture["match_id"]] = checked
    if news_provider is not None and (news_provider not in {"bing_rss", "bing_news_rss"} or not verified):
        raise ValueError("News acquisition requires supported transport and independently verified identities")
    for policy in policies:
        if not parse_time(policy.model.frozen_at) <= now < parse_time(policy.model.cutoff):
            raise ValueError("Initialization is from the future or its prematch horizon has expired")
        verify_prior_archive(policy.manifest, archive_root)
    f_only_policies, f_only_bytes = {}, None
    if f_only_predictor_path is not None:
        from football_agent.historical_f_only_policy import HistoricalFOnlyPolicy, fixture_from_verified_identity
        from football_research.acquisition.historical_context_training import HistoricalContextModel
        f_only_bytes = Path(f_only_predictor_path).read_bytes()
        semantic_spec = None
        if f_only_semantic_config_path is not None:
            from football_research.acquisition.entity_semantic_f import EntitySemanticF
            from football_agent.entity_semantic_f_only_policy import EntitySemanticFOnlyPolicy, validate_config
            semantic_spec = validate_config(json.loads(Path(f_only_semantic_config_path).read_bytes()), f_only_bytes)
            f_only_model = EntitySemanticF.load(f_only_predictor_path)
        else:
            f_only_model = HistoricalContextModel.load(f_only_predictor_path)
        for base in policies:
            mid = base.model.fixture["match_id"]
            policy_type = EntitySemanticFOnlyPolicy if semantic_spec is not None else HistoricalFOnlyPolicy
            replay_trace = None
            if f_only_replay_traces is not None:
                replay_trace = f_only_replay_traces.get(mid, f_only_replay_traces.get("__single__"))
                if replay_trace is None:
                    raise ValueError("Entity semantic replay has no trace for selected match")
            runtime_f = policy_type(base, f_only_model,
                fixture=fixture_from_verified_identity(base, verified[mid]), artifact=f_only_bytes,
                **({"semantic_config": semantic_spec, "replay_trace": replay_trace}
                   if semantic_spec is not None else {}))
            runtime_f.check_start(now)
            f_only_policies[mid] = runtime_f
    context_calibration, context_checked, context_raw = None, None, None
    context_confirmation, confirmation_spec, value_raw = None, None, None
    if context_registration_path is not None:
        from .context_collection import validate_context_collection, verify_context_run
        context_calibration = {
            "registration": json.loads(Path(context_registration_path).read_text(encoding="utf-8")),
            "initialization_plan": plan, "identity_bundle": {"bindings": verified},
            "previous_plans": [json.loads(Path(p).read_text(encoding="utf-8"))
                               for p in previous_learning_plan_paths]}
        context_checked = validate_context_collection(context_calibration,
            predictor_path=context_predictor_path, as_of=now)
        context_raw = Path(context_predictor_path).read_bytes()
    if context_confirmation_path is not None:
        from .context_collection import confirmation_policy_spec, verify_context_confirmation_run
        from football_research.acquisition.context_confirmation import validate_context_confirmation
        confirmation_spec = confirmation_policy_spec(seed=seed, requests_per_match=requests_per_match,
            source_profile_version=source_profile_version, news_projection_version=news_projection_version,
            evidence_version=evidence_version, candidate_discovery_version=candidate_discovery_version,
            news_provider=news_provider, route=route, news_route_overrides=news_route_overrides,
            allowed_hosts=allowed_hosts, seed_urls=seed_urls, base_policy=policies[0])
        if any(p.manifest['loss'] != confirmation_spec['loss'] or
               p.lambda_cost != confirmation_spec['lambda_cost'] for p in policies):
            raise ValueError('Confirmation batch requires its registered loss and cost conversion')
        context_confirmation = {
            'registration': json.loads(Path(context_confirmation_path).read_text(encoding='utf-8')),
            'initialization_plan': plan, 'identity_bundle': {'bindings': verified},
            'previous_plans': [json.loads(Path(p).read_text(encoding='utf-8')) for p in previous_learning_plan_paths]}
        context_checked = validate_context_confirmation(context_confirmation, predictor_path=context_predictor_path,
            value_model_path=context_value_model_path, as_of=now, policy_spec=confirmation_spec)
        context_raw = Path(context_predictor_path).read_bytes()
        value_raw = Path(context_value_model_path).read_bytes()
    learning_context, learning, artifact_bytes = None, None, {}
    if learning_plan_path is not None:
        from .prospective_learning import validate_learning_plan
        learning_context = {
            "learning_plan": json.loads(Path(learning_plan_path).read_text(encoding="utf-8")),
            "initialization_plan": plan, "identity_bundle": {"bindings": verified},
            "previous_plans": [json.loads(Path(p).read_text(encoding="utf-8"))
                               for p in previous_learning_plan_paths]}
        learning = validate_learning_plan(learning_context["learning_plan"],
            initialization_plan=plan, identity_bundle=learning_context["identity_bundle"],
            previous_plans=learning_context["previous_plans"], as_of=now)
        for key, entry in learning_context["learning_plan"]["artifacts"].items():
            raw = Path(entry["path"]).read_bytes()
            if sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Learning artifact changed before collection snapshot")
            artifact_bytes[key] = raw
    observable_shadow_spec, observable_shadow_bytes = None, None
    if observable_shadow_path is not None:
        if context_mode or learning_plan_path is not None:
            raise ValueError("Observable shadow opt-in is isolated to non-g market-prior collection")
        from football_agent.observable_prequery_shadow import load_shadow_spec
        observable_shadow_spec, observable_shadow_bytes = load_shadow_spec(observable_shadow_path)
    output.mkdir(parents=True)
    if observable_shadow_spec is not None:
        from football_spider.storage.raw_archive import atomic_write
        from football_agent.observable_prequery_shadow import MODEL_FILE
        atomic_write(output / MODEL_FILE, observable_shadow_bytes)
    if context_calibration is not None:
        from football_spider.storage.raw_archive import atomic_write
        _save(output/"context-calibration.json", context_calibration)
        atomic_write(output/"context-predictor.json", context_raw)
    if context_confirmation is not None:
        from football_spider.storage.raw_archive import atomic_write
        _save(output/'context-confirmation.json', context_confirmation)
        atomic_write(output/'context-predictor.json', context_raw)
        atomic_write(output/'context-value-model.json', value_raw)
    if learning_context is not None:
        from football_spider.storage.raw_archive import atomic_write
        _save(output / "learning-context.json", learning_context)
        for key, raw in artifact_bytes.items():
            atomic_write(output / ("learning-" + key.replace("_", "-") + ".json"), raw)
    # A frozen policy and sampling seed are reusable configuration, not the
    # identity of a new acquisition. Persist a nonce before any web execution;
    # even simultaneous calls at the same clock must produce distinct run IDs.
    collection_id = "prospective-batch-" + uuid4().hex
    batch_config = {"collection_id": collection_id, "plan_sha256": sha256(plan_bytes).hexdigest(),
        "started_at": now.isoformat(), "source_capabilities": {"allowed_hosts": list(allowed_hosts), "seed_urls": list(seed_urls)},
        "requests_per_match": requests_per_match, "seed": seed, "route": route,
        "maximum_http_requests": len(policies) * requests_per_match,
        "selection_basis": ("existing_uniform_sampler_over_current_unknown_menu" if policy_arm == "voi"
            else {"fixed_order": "fixed_order_predeclared_menu_order", "full_probe": "full_probe_predeclared_menu_order",
                  "random_explore": "random_explore_predeclared_uniform"}[policy_arm]),
        "policy_selection_basis": ({"voi": "not_configured_or_bounded_unknown_exploration",
            "random_explore": "random_explore_predeclared_uniform",
            "fixed_order": "fixed_order_predeclared_menu_order",
            "full_probe": "full_probe_predeclared_menu_order"}[policy_arm]),
        "policy_arm": policy_arm, "policy_arm_version": policy_arm_version,
        "arm_assigned_at": arm_assigned_at, "random_seed": arm_seed_base,
        "propensity_logging": propensity_logging,
        "arm_assignments": deepcopy(arm_assignments),
        "information_mode": "prospective_capture",
        "arm_selection_design": {"order": "generated_menu_order",
            "finite_feasible_candidate_sequence": True,
            "unknown_value_not_estimated": True}}
    if training_exploration_spec is not None:
        batch_config["training_exploration"] = {
            "schema": training_exploration_spec["schema"],
            "enabled": True,
            "epsilon": float(training_exploration_spec.get("epsilon", 0.2)),
            "seed": exploration_seed,
            "selection_authority": "data_collection_only",
            "empirical_g_enabled": False,
            "f_invoked_by_adapter": False,
            "label_status": "pending_outcome",
        }
    if verified:
        batch_config.update(identity_bindings=verified, news_provider=news_provider,
                            identity_archive_scope="independent_fixture_setup_not_policy_queries")
    if news_route_overrides is not None:
        batch_config['news_route_overrides'] = deepcopy(news_route_overrides)
    if source_profile_version >= 2:
        batch_config['source_profile_version'] = source_profile_version
    if news_projection_version >= 2:
        batch_config['news_projection_version'] = news_projection_version
    if evidence_version is not None:
        batch_config['evidence_version'] = evidence_version
    if candidate_discovery_version is not None:
        batch_config['candidate_discovery_version'] = candidate_discovery_version
    if learning is not None:
        batch_config["learning_cohort"] = learning["binding"]
        batch_config["learning_context_sha256"] = sha256((output / "learning-context.json").read_bytes()).hexdigest()
        if learning["text_model"] is not None:
            batch_config["selection_basis"] = "frozen_text_policy_with_bounded_unknown_exploration"
    if context_calibration is not None:
        batch_config["context_calibration"] = context_checked["binding"]
        batch_config["context_calibration_sha256"] = sha256((output/"context-calibration.json").read_bytes()).hexdigest()
        batch_config["selection_basis"] = "frozen_context_f_with_bounded_unknown_exploration"
        batch_config["evidence_origin"] = context_checked["predictor"].metadata["evidence_origin"]
    if context_confirmation is not None:
        batch_config['context_confirmation'] = context_checked['binding']
        batch_config['context_confirmation_sha256'] = sha256((output/'context-confirmation.json').read_bytes()).hexdigest()
        batch_config['selection_basis'] = 'frozen_context_f_g_with_bounded_unknown_exploration'
        batch_config['evidence_origin'] = context_checked['predictor'].metadata['evidence_origin']
    if journal_enabled:
        batch_config["acquisition_journal"] = {"enabled": True,
            "adapter": "football_research.data_adapters.prospective_journal",
            "model_status": "PREDICTOR_ONLY" if f_only_policies else "NOT_CONFIGURED",
            "raw_context_capture": True}
    if f_only_policies:
        batch_config["f_only_predictor"] = {"sha256": sha256(f_only_bytes).hexdigest(),
            "model_version": f_only_model.model_version, "bundled_g_enabled": False,
            "probability_basis": next(iter(f_only_policies.values())).manifest["probability_basis"],
            "calibration_status": "not_established"}
        if f_only_semantic_config_path is not None:
            batch_config["f_only_predictor"]["semantic_config"] = deepcopy(semantic_spec)
        if f_only_replay_bytes is not None:
            batch_config["f_only_execution"] = {
                "mode": "codex_typed_replay",
                "provider_http": 0,
                "deepseek_invoked": False,
                "replay_trace_sha256": sha256(f_only_replay_bytes).hexdigest(),
                "replay_trace_path_scope": "caller_supplied_frozen_input",
                "selection_authority": "controlled_exploration_only",
            }
    _save(output / "batch-configuration.json", batch_config)
    source_archive = RawArchive(Path(archive_root))
    runs = []
    for index, policy in enumerate(policies):
        assignment = arm_assignments[index]
        current_arm = assignment["policy_arm"]
        current_arm_version = assignment["policy_arm_version"]
        current_assigned_at = assignment["arm_assigned_at"]
        current_random_seed = assignment["random_seed"]
        current_propensity_logging = assignment["propensity_logging"]
        runtime_policy = f_only_policies.get(policy.model.fixture["match_id"], policy)
        if context_calibration is not None:
            from football_agent.context_voi_policy import ContextVOIPolicy
            context_checked = validate_context_collection(context_calibration,
                predictor_path=output/"context-predictor.json", as_of=datetime.now(timezone.utc))
            runtime_policy = ContextVOIPolicy(policy, context_checked["predictor"],
                physical_match_id=context_checked["matches"][policy.model.fixture["match_id"]]["source_event_id"],
                predictor_artifact=(output/"context-predictor.json").read_bytes())
        if context_confirmation is not None:
            from football_agent.context_voi_policy import ContextVOIPolicy
            context_checked = validate_context_confirmation(context_confirmation,
                predictor_path=output/'context-predictor.json', value_model_path=output/'context-value-model.json',
                as_of=datetime.now(timezone.utc), policy_spec=confirmation_spec)
            runtime_policy = ContextVOIPolicy(policy, context_checked['predictor'],
                physical_match_id=context_checked['matches'][policy.model.fixture['match_id']]['source_event_id'],
                value_model=context_checked['value_model'], predictor_artifact=(output/'context-predictor.json').read_bytes(),
                value_artifact=(output/'context-value-model.json').read_bytes())
        if learning_context is not None:
            # Recheck the actual start clock for each match, using the immutable
            # local snapshots rather than mutable training artifact locations.
            learning = validate_learning_plan(learning_context["learning_plan"],
                initialization_plan=plan, identity_bundle=learning_context["identity_bundle"],
                previous_plans=learning_context["previous_plans"], as_of=datetime.now(timezone.utc),
                artifact_paths={key: output / ("learning-" + key.replace("_", "-") + ".json")
                                for key in artifact_bytes})
            if learning["text_model"] is not None:
                from football_agent.text_voi_policy import TextVOIPolicy
                runtime_policy = TextVOIPolicy(policy, learning["text_model"], risk_model=learning["risk_model"])
        match = policy.model.fixture
        folder = output / ("match-" + match["match_id"].rsplit(":", 1)[-1])
        identity = {"match_id": match["match_id"], "home_team": match["home_team_name"],
            "away_team": match["away_team_name"], "kickoff_at": match["kickoff"],
            "prediction_cutoff": policy.model.cutoff}
        binding = verified.get(match["match_id"])
        run_seeds, run_hosts = list(seed_urls), list(allowed_hosts)
        index_urls = []
        if binding:
            from football_agent.evidence_candidates import EvidenceDocumentCandidates
            from football_agent.match_source_profiles import resolve_verified_profile
            profile = resolve_verified_profile(binding, version=source_profile_version)
            run_seeds = sorted(set(run_seeds + profile["seed_urls"]))
            run_hosts = sorted(set(run_hosts + profile["allowed_hosts"]))
            index_urls = [entry["url"] for entry in profile["entries"]
                          if entry["kind"] in {"news_index", "club_home"}]
            identity.update(home_aliases=binding["aliases"]["home"], away_aliases=binding["aliases"]["away"])
            if source_profile_version in {6, 7, 8, 9}:
                # Raw source names remain in the market initialization. Only
                # this new official profile opts the English source discovery
                # and grounded follow-ups into verified English names.
                identity.update(home_team=binding["canonical"]["home_name"],
                                away_team=binding["canonical"]["away_name"])
            generator = EvidenceDocumentCandidates(
                match_terms=[binding["canonical"][side + "_name"] for side in ("home", "away")],
                seed_urls=run_seeds, allowed_hosts=run_hosts, numeric_actions=[],
                web_cost={"money": .01, "requests": 1.}, match=identity, profile=profile,
                discovery_version=(candidate_discovery_version if candidate_discovery_version is not None
                                   else {1: 2, 2: 3, 3: 4, 4: 10, 5: 11, 6: 12, 7: 12, 8: 13, 9: 14}[source_profile_version]))
        else:
            generator = MatchDocumentCandidates(match_terms=[identity["home_team"], identity["away_team"]],
                seed_urls=run_seeds, allowed_hosts=run_hosts, numeric_actions=[],
                web_cost={"money": .01, "requests": 1.})
        if observable_shadow_spec is not None:
            from football_agent.observable_prequery_shadow import ObservablePrequeryShadow, MODEL_FILE
            generator = ObservablePrequeryShadow(generator,
                registry=observable_shadow_spec["registry"], model_bytes=observable_shadow_bytes,
                expected_model_sha256=observable_shadow_spec["model_raw_sha256"])
        if news_provider:
            from football_agent.news_discovery import NewsDiscoveryTools
            tools = NewsDiscoveryTools(RawArchive(folder / "source-archive"), run_hosts,
                provider=news_provider, default_route=route, extraction_version=3,
                route_overrides=news_route_overrides,
                index_urls=index_urls, projection_version=news_projection_version)
        else:
            tools = RoutedOpenWebTools(RawArchive(folder / "source-archive"), run_hosts,
                                        route=route, extraction_version=3)
        # The shared executor constructs an empty ControlledEpisode even when
        # numeric_actions=[]; its clock refers only to the known free context.
        # No feature values or numeric acquisition capabilities are supplied.
        row = {**match, "prediction_cutoff": policy.model.cutoff,
               "features_available_at": match["fixture_available_at"]}
        executor = ArchivedAndWebExecutor(row, [], tools, seed_urls=run_seeds)
        run_key = {"collection_id": collection_id, "model": policy.manifest, "seed": seed + index,
                   "policy_arm": current_arm, "policy_arm_version": current_arm_version,
                   "random_seed": current_random_seed}
        if binding:
            run_key.update(generator=generator.configuration(), transport=tools.configuration())
        run_id = "prospective-text-" + digest(run_key)[:24]
        executor.run_id = run_id
        journal_executor = None
        if journal_enabled:
            from .prospective_journal import JournaledProspectiveExecutor
            journal_executor = JournaledProspectiveExecutor(executor,
                root=output / "acquisition-journal", fixture=match, request_limit=requests_per_match,
                policy_arm=current_arm, policy_version=runtime_policy.version,
                random_seed=current_random_seed, propensity_logging=current_propensity_logging,
                **({"predictor_policy": runtime_policy} if f_only_policies else {}))
        limits = {"money": .01 * requests_per_match, "requests": requests_per_match}
        runtime_seed = current_random_seed
        runtime = LearnedVOIRuntime(runtime_policy, seed=runtime_seed, random_seed=runtime_seed,
            policy_arm=current_arm, policy_arm_version=current_arm_version,
            arm_assigned_at=current_assigned_at, propensity_logging=current_propensity_logging,
            exploration_limits=limits, max_exploration_actions=requests_per_match)
        training_exploration = None
        if training_exploration_spec is not None:
            from football_agent.training_exploration_adapter import TrainingExplorationAdapter
            adapter_seed = training_exploration_spec.get("seed", seed) + index
            training_exploration = TrainingExplorationAdapter(
                enabled=True,
                epsilon=float(training_exploration_spec.get("epsilon", 0.2)),
                seed=int(adapter_seed))
        metadata = {"collection_id": collection_id,
            "generator": generator.configuration(), "transport": tools.configuration(),
            "fixture": {"match_id": match["match_id"], "match": identity["home_team"] + " vs " + identity["away_team"],
                "source_score_period": match["score_period"], "forecast_target_period": "regulation",
                "fixture_identity_scope": "current_source_snapshot_not_independent_confirmation"},
            "information_mode": "prospective_capture",
            "policy_arm": current_arm,
            "policy_arm_version": current_arm_version,
            "arm_assigned_at": current_assigned_at,
            "random_seed": current_random_seed,
            "propensity_logging": current_propensity_logging,
            "input_bindings": {"initialization_plan_sha256": sha256(plan_bytes).hexdigest(),
                "model_version": policy.model.model_version, "market_prior_sha256": _digest(policy.model.binding),
                "initialization_mode": "operational", "prior_archive": "prior-archive"}}
        if journal_executor is not None:
            metadata["acquisition_journal"] = {"run_id": journal_executor.run_id,
                "root": "acquisition-journal", "path_scope": "batch_relative",
                "model_status": "NOT_CONFIGURED", "prediction": None,
                "candidate_expected_gains": None}
            if f_only_policies:
                metadata["acquisition_journal"].update(model_status="PREDICTOR_ONLY",
                    predictor_version=runtime_policy.predictor.model_version,
                    predictor_sha256=runtime_policy.predictor_sha256,
                    bundled_g_enabled=False)
        if training_exploration is not None:
            metadata["training_exploration"] = {
                "adapter_schema": "footballmind-g-training-exploration-adapter-v1",
                "selection_authority": "data_collection_only",
                "empirical_g_enabled": False,
                "f_invoked_by_adapter": False,
                "label_status": "pending_outcome",
            }
        if f_only_replay_bytes is not None:
            metadata["f_only_execution"] = {
                "mode": "codex_typed_replay",
                "provider_http": 0,
                "deepseek_invoked": False,
                "replay_trace_sha256": sha256(f_only_replay_bytes).hexdigest(),
                "selection_authority": "controlled_exploration_only",
            }
        if binding:
            metadata["verified_fixture_identity"] = binding
            metadata["fixture"]["fixture_identity_scope"] = "raw_verified_independent_official_fixture"
            metadata["input_bindings"]["identity_binding_sha256"] = binding["binding_sha256"]
        if learning is not None:
            metadata["learning_cohort"] = learning["by_match"][match["match_id"]]
            metadata["input_bindings"]["learning_context_sha256"] = batch_config["learning_context_sha256"]
        if context_calibration is not None:
            metadata["context_calibration"] = context_checked["binding"]
            metadata["input_bindings"]["context_calibration_sha256"] = batch_config["context_calibration_sha256"]
        if context_confirmation is not None:
            metadata['context_confirmation'] = context_checked['binding']
            metadata['input_bindings']['context_confirmation_sha256'] = batch_config['context_confirmation_sha256']
        def record_menu(payload, *, _archive=tools.archive, _match=match, _run_id=run_id):
            # Persist the exact pre-I/O menu through the existing RawArchive.
            # The payload contains candidates and their order, not hidden
            # provider returns or a post-hoc reconstruction.
            if observable_shadow_bytes is not None and not (folder / MODEL_FILE).exists():
                # Runtime has created its directory; pin the local model before I/O.
                atomic_write(folder / MODEL_FILE, observable_shadow_bytes)
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
            observed = parse_time(payload["observed_at"])
            obs = _archive.save("agent_trajectory",
                "https://footballmind.local/agent-trajectory/menu",
                {"run_id": _run_id, "match_id": _match["match_id"], "step": payload["step"]},
                raw, status=200, headers={"content-type": "application/json"},
                context={"kind": "candidate_menu", "information_mode": "prospective_capture"},
                observed_at=observed)
            if journal_executor is not None:
                journal_executor.observe_menu(payload)
            return {"menu_raw_path": obs.payload_path, "menu_raw_sha256": obs.content_hash,
                    "menu_raw_observation_id": obs.id}

        def record_context_raw(payload, *, _archive=tools.archive, _match=match, _run_id=run_id):
            captured = datetime.now(timezone.utc)
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
            obs = _archive.save("agent_trajectory",
                "https://footballmind.local/agent-trajectory/context",
                {"run_id": _run_id, "match_id": _match["match_id"],
                 "snapshot_kind": payload["snapshot_kind"], "step": payload.get("step")},
                raw, status=200, headers={"content-type": "application/json"},
                context={"kind": "runtime_context", "information_mode": "prospective_capture"},
                observed_at=captured)
            saved_at = datetime.now(timezone.utc).isoformat()
            if journal_executor is not None and f_only_policies:
                journal_executor.observe_context({**payload, "raw_path": obs.payload_path,
                    "raw_sha256": obs.content_hash, "raw_observation_id": obs.id})
            return {"captured_at": saved_at,
                    "archive_save_started_at": captured.isoformat(),
                    "archive_save_returned_at": saved_at,
                    "capture_timing_basis": "measured_local_utc_archive_save_return",
                    "persistence_boundary": "return_observed_fsync_not_instrumented", "raw_path": obs.payload_path,
                    "raw_sha256": obs.content_hash, "raw_observation_id": obs.id}

        runtime_options = {"context_recorder": record_context_raw} if journal_enabled else {}
        try:
            result = runtime.run(policy.model.initial, run_id=run_id, limits=limits,
                propose=generator, execute=journal_executor or executor, output=folder,
                provider_version=executor.version + ":" + digest(tools.configuration()),
                generator_version=generator.version + ":" + digest(generator.configuration()),
                run_metadata=metadata, evidence_match=identity, evidence_version=evidence_version,
                trajectory_recorder=record_menu, training_exploration=training_exploration,
                **runtime_options)
        except Exception as exc:
            if f_only_semantic_config_path is not None:
                # Retain attempts and reservations; never retry a failed run
                # against live sources. finish reconciles any pending request.
                summary = journal_executor.finish("entity_semantic_runtime_failed")
                failure = {"schema": "entity-semantic-collection-failure-v1", "run_id": run_id,
                    "error_type": type(exc).__name__, "journal": summary,
                    "numeric_voi_training_eligible": False, "automatic_retry_allowed": False}
                folder.mkdir(parents=True, exist_ok=True)
                _save(folder / "collection-failure.json", failure)
            raise
        journal_summary = (journal_executor.finish(result["terminal_selection"].get("reason", "runtime_stopped"))
                           if journal_executor is not None else None)
        if context_calibration is not None:
            atomic_write(folder/"context-calibration.json", (output/"context-calibration.json").read_bytes())
            verify_context_run(folder, json.loads((folder/"configuration.json").read_text(encoding="utf-8")))
        if context_confirmation is not None:
            atomic_write(folder/'context-confirmation.json', (output/'context-confirmation.json').read_bytes())
            verify_context_confirmation_run(folder, json.loads((folder/'configuration.json').read_text(encoding='utf-8')))
        if learning_context is not None:
            atomic_write(folder / "learning-context.json", (output / "learning-context.json").read_bytes())
            for key, raw in artifact_bytes.items():
                atomic_write(folder / ("learning-" + key.replace("_", "-") + ".json"), raw)
        copied = RawArchive(folder / "prior-archive")
        for obs in source_archive.observations('spftll'):
            if obs.observed_at <= parse_time(policy.model.frozen_at):
                copied.save(obs.source, obs.endpoint, obs.params, source_archive.read_body(obs),
                    status=obs.http_status, headers=obs.response_headers, request_id=obs.request_id,
                    context=obs.context, error_class=obs.error_class, observed_at=obs.observed_at, observation_id=obs.id)
        verify_prior_archive(policy.manifest, copied.root)
        if binding:
            _copy_observations(RawArchive(Path(identity_archive)), RawArchive(folder / "identity-archive"),
                observation_ids={entry["observation_id"] for entry in binding["observations"]})
            verify_run_identity(folder, json.loads((folder / "configuration.json").read_text(encoding="utf-8")))
        if learning_context is not None:
            verify_run_learning(folder, json.loads((folder / "configuration.json").read_text(encoding="utf-8")))
        _save(folder / "market-prior.json", policy.model.binding)
        _save(folder / "prior-bindings.json", {policy.model.model_version: policy.prior_binding()})
        _save(folder / "settlements.json", [])
        write_analysis_report(result, match=identity, output=folder)
        runs.append({"match_id": match["match_id"], "folder": str(folder.resolve()),
            "run_id": run_id, "initial_probabilities": result["initial_probabilities"],
            "http_requests": result["measured_resources"]["requests"],
            "attempts": len(result["attempts"]), "documents": len(result["documents"]),
            "terminal_selection": result["terminal_selection"], "labels": 0,
            **({"acquisition_journal": journal_summary} if journal_summary is not None else {})})
        _save(output / "batch-progress.json", {"runs": runs, "planned_matches": len(policies)})
    report = {"status": "collected_pending_independent_settlement", "runs": runs,
        "unique_matches": len(runs), "http_requests": sum(r["http_requests"] for r in runs),
        "independent_labels": 0, "numeric_voi_training_eligible": 0,
        "realized_voi_benefit_proven": False, "initialization_plan_sha256": sha256(plan_bytes).hexdigest()}
    _save(output / "batch-report.json", report)
    return report


def _verified_result_rows(run_folders, *, result_archive, as_of):
    """Derive official results again from raw bytes; an attestation is not a score."""
    from .verified_fixtures import verify_independent_settlement
    from football_spider.storage.raw_archive import RawArchive
    observation_ids = ([obs.id for obs in RawArchive(Path(result_archive)).observations()]
                       if result_archive is not None else [])
    checks, labels = [], []
    for folder in map(Path, run_folders):
        config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        binding = verify_run_identity(folder, config)
        if binding is None:
            raise ValueError("Raw-verified result audit requires independently bound run identities")
        if result_archive is None:
            check = {"status": "PENDING", "reason": "independent_result_archive_missing", "settlement_rows": []}
        else:
            check = verify_independent_settlement(binding,
                identity_archive_root=folder / "identity-archive", result_archive_root=result_archive,
                observation_ids=observation_ids, as_of=as_of)
        checks.append({"match_id": config["initial_state"]["match_id"], **check})
        if check["status"] == "VERIFIED":
            labels.extend(check["settlement_rows"])
        elif check.get("settlement_rows"):
            raise ValueError("Unverified results cannot supply training labels")
    return labels, checks


def join_settlements(run_folders, *, settlements, prior_bindings, as_of, result_archive=None):
    """Reuse existing independently normalised result sidecars and raw audit."""
    from .open_text_episodes import build_open_text_episodes, training_records
    run_folders, settlements = list(run_folders), list(settlements)
    verified_folders = [folder for folder in run_folders
        if json.loads((Path(folder)/"configuration.json").read_text(encoding="utf-8"))
        .get("supervisor_metadata", {}).get("verified_fixture_identity") is not None]
    result_checks = []
    if verified_folders:
        if settlements and result_archive is None:
            raise ValueError("Verified prospective runs require original result bytes, not label-only sidecars")
        rows, result_checks = _verified_result_rows(verified_folders, result_archive=result_archive, as_of=as_of)
        ids = {row["match_id"] for row in result_checks}
        external = [row for row in settlements if row.get("match_id") in ids]
        if external and sorted(map(_canonical, external)) != sorted(map(_canonical, rows)):
            raise ValueError("Supplied results differ from the independently reconstructed official scores")
        settlements = [row for row in settlements if row.get("match_id") not in ids] + rows
    dataset = build_open_text_episodes(run_folders, settlements=settlements,
                                      prior_bindings=prior_bindings, as_of=as_of)
    result = {"dataset": dataset, "training_records": training_records(dataset)}
    if verified_folders:
        result["independent_result_verification"] = result_checks
    return result


def main(argv=None):
    """Prepare and collect now, then attach independently exported results later."""
    import argparse
    from .spftll import load_spftll_archive
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=3)
    prepare.add_argument("--horizon-hours", type=float, default=48,
                         help="Future window in hours, at most 1008 (42 days); default preserves 48 hours")
    prepare.add_argument("--fixture-id", action="append",
                         help="Exact predeclared match ID; repeat to restrict the frozen selection")
    collect = commands.add_parser("collect")
    collect.add_argument("--archive", type=Path, required=True)
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--allowed-host", action="append", required=True)
    collect.add_argument("--seed-url", action="append", default=[])
    collect.add_argument("--requests-per-match", type=int, default=2)
    collect.add_argument("--seed", type=int, default=179)
    collect.add_argument("--policy-arm", choices=["voi", "fixed_order", "random_explore", "full_probe"],
                         default="voi")
    collect.add_argument("--policy-arm-version", default="footballmind-policy-arm-v1")
    collect.add_argument("--arm-assigned-at", help="Frozen ISO-8601 arm assignment time")
    collect.add_argument("--random-seed", type=int, help="Precommitted arm randomization seed")
    collect.add_argument("--arm-schedule", type=Path,
                         help="JSON list with one predeclared policy-arm assignment per selected match")
    collect.add_argument("--no-propensity-logging", action="store_false", dest="propensity_logging",
                         help="Do not persist randomized selection probabilities")
    collect.set_defaults(propensity_logging=True)
    collect.add_argument("--journal-enabled", action="store_true",
                         help="Record existing AcquisitionDecisionJournal intents and raw context snapshots")
    collect.add_argument("--f-only-predictor", type=Path,
                         help="Frozen text f with journal and controlled exploration; add semantic config for entity f")
    collect.add_argument("--route", choices=["environment", "direct"], default="environment")
    collect.add_argument("--identity-bundle", type=Path)
    collect.add_argument("--identity-archive", type=Path)
    collect.add_argument("--learning-plan", type=Path,
                         help="Pre-acquisition cohort and frozen model commitment")
    collect.add_argument("--news-route-overrides", type=Path, help="Explicit host to transport route JSON; no automatic failover")
    collect.add_argument("--context-registration", type=Path)
    collect.add_argument("--context-predictor", type=Path)
    collect.add_argument("--f-only-semantic-config", type=Path,
                         help="Explicit entity shadow prior/provider binding; omitted preserves historical f-only")
    collect.add_argument("--f-only-replay-trace", type=Path,
                         help="Opt-in frozen typed-v3 extraction trace; bypasses provider execution and network")
    collect.add_argument("--training-exploration", type=Path,
                         help="Opt-in g training exploration config; preserves old selection when omitted")
    collect.add_argument("--context-confirmation", type=Path, help="Independent C registration; mutually exclusive with B")
    collect.add_argument("--context-value-model", type=Path, help="Frozen g artifact bound by C registration")
    collect.add_argument("--previous-learning-plan", action="append", type=Path, default=[])
    collect.add_argument("--news-provider", choices=["bing_rss", "bing_news_rss"])
    collect.add_argument("--source-profile-version", type=int, choices=list(range(1, 10)), default=1)
    collect.add_argument("--news-projection-version", type=int, choices=list(range(1, 14)), default=1)
    collect.add_argument("--candidate-discovery-version", type=int, choices=list(range(1, 15)),
                         help="Explicit candidate version; omitted preserves the source-profile mapping")
    from football_agent.match_evidence import SUPPORTED_VERSIONS
    collect.add_argument("--evidence-version", choices=SUPPORTED_VERSIONS,
                         help="Explicit evidence projection; omitted preserves the existing default")
    audit = commands.add_parser("audit")
    audit.add_argument("--batch", type=Path, required=True)
    audit.add_argument("--settlements", type=Path)
    audit.add_argument("--result-archive", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("The output already exists; choose a new immutable output")
    if args.command == "prepare":
        now = datetime.now(timezone.utc)
        result = select_initializations(load_spftll_archive(args.archive / "raw", as_of=now),
                                        as_of=now, limit=args.limit, horizon_hours=args.horizon_hours,
                                        fixture_ids=args.fixture_id)
        _save(args.output, result)
        print(json.dumps({"selected_matches": len(result["selected"]), "output": str(args.output.resolve())}))
    elif args.command == "collect":
        result = collect_prepared_batch(args.plan, output=args.output, archive_root=args.archive,
            allowed_hosts=args.allowed_host, seed_urls=args.seed_url,
            requests_per_match=args.requests_per_match, seed=args.seed, route=args.route,
            policy_arm=args.policy_arm, policy_arm_version=args.policy_arm_version,
            arm_assigned_at=args.arm_assigned_at, random_seed=args.random_seed,
            propensity_logging=args.propensity_logging,
            journal_enabled=args.journal_enabled,
            f_only_predictor_path=args.f_only_predictor,
            f_only_semantic_config_path=args.f_only_semantic_config,
            f_only_replay_trace_path=args.f_only_replay_trace,
            training_exploration_path=args.training_exploration,
            arm_schedule=(json.loads(args.arm_schedule.read_text(encoding="utf-8"))
                          if args.arm_schedule else None),
            identity_bundle=args.identity_bundle, identity_archive=args.identity_archive,
            news_provider=args.news_provider, source_profile_version=args.source_profile_version,
            news_projection_version=args.news_projection_version, evidence_version=args.evidence_version,
            learning_plan_path=args.learning_plan, previous_learning_plan_paths=args.previous_learning_plan,
            candidate_discovery_version=args.candidate_discovery_version,
            context_registration_path=args.context_registration, context_predictor_path=args.context_predictor,
            context_confirmation_path=args.context_confirmation, context_value_model_path=args.context_value_model,
            news_route_overrides=(json.loads(args.news_route_overrides.read_text(encoding="utf-8"))
                                  if args.news_route_overrides else None))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        folders = sorted(args.batch.glob("match-*"))
        proofs = {}
        for folder in folders:
            proofs.update(json.loads((folder / "prior-bindings.json").read_text(encoding="utf-8")))
        labels = json.loads(args.settlements.read_text(encoding="utf-8")) if args.settlements else []
        if not isinstance(labels, list):
            raise ValueError("Settlements must be existing export_results rows, never guessed outcomes")
        result = join_settlements(folders, settlements=labels, prior_bindings=proofs,
                                 as_of=datetime.now(timezone.utc), result_archive=args.result_archive)
        _save(args.output, result)
        print(json.dumps(result["dataset"]["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
