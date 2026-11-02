"""Offline wiring checks against the frozen P2 model and actual P3 snapshots."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.historical_context_policy import HistoricalContextVOIPolicy
from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.evaluation.frozen_period import load_frozen_models


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "var/submission/p3-single-match/live-01"
MODEL = ROOT / "var/submission/p2-historical-context/real-article-panels-v1/model.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Policy checks must use frozen artifacts without network access")
    for obj, name in ((socket, "getaddrinfo"), (socket, "create_connection"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, blocked)


@pytest.fixture(scope="module")
def frozen():
    freeze_path = ROOT / "var/parallel/agent/round175/frozen-next-period.json"
    paths = [MODEL, RUN / "run.json", RUN / "configuration.json",
             RUN / "context-predictor.json", RUN / "context-value-model.json", freeze_path]
    if any(not p.is_file() for p in paths):
        pytest.skip("The separately preserved real P2/P3 experiment artifacts are absent")
    artifact = MODEL.read_bytes()
    assert artifact == paths[3].read_bytes() == paths[4].read_bytes()
    config = json.loads(paths[2].read_text(encoding="utf-8"))
    manifest = config["policy_manifest"]
    assert sha256(artifact).hexdigest() == manifest["predictor_sha256"]
    _, base_model = load_frozen_models(ROOT, json.loads(freeze_path.read_text(encoding="utf-8")),
                                      design_path=ROOT / "docs/research/acquisition/round175-design.md")
    spec = manifest["base_policy"]
    base = LearnedJointVOIPolicy(base_model, lambda_cost=spec["lambda_cost"], loss=spec["loss"],
        min_effective_matches=spec["min_effective_matches"], min_class_weight=spec["min_class_weight"])
    model = HistoricalContextModel.load(MODEL)
    policy = HistoricalContextVOIPolicy(base, model, fixture=manifest["fixture"], artifact=artifact)
    assert policy.manifest == manifest
    return policy, json.loads(paths[1].read_text(encoding="utf-8"))["cycles"]


def plan(policy, cycle, public=None, actions=None):
    public = cycle["public_before"] if public is None else public
    actions = cycle["generated_actions"] if actions is None else actions
    return policy.plan_public(public, actions, public["budget"],
        candidate_generation_complete=cycle["strict_decision"]["candidate_generation_complete"])


def test_real_body_and_observed_anchors_reach_the_frozen_model(frozen):
    policy, cycles = frozen
    public = cycles[-1]["public_before"]
    state, visible, skipped = policy._visible(public, cycles[-1]["generated_actions"])
    article = next(d for d in public["documents"] if d.get("kind") == "article")
    modeled = next(d for d in state["documents"] if d["url"] == article["url"])
    assert modeled["paragraphs"]
    assert modeled["text"] == "\n".join(modeled["paragraphs"])
    assert all(p in article["text"] for p in modeled["paragraphs"])
    assert modeled["raw_sha256"] == article["raw_sha256"]
    assert modeled["source_published_at"] == article["source_published_at"]
    assert state["observed_links"] and all(link["title"] for link in state["observed_links"])
    assert not skipped and not visible["audit"]["excluded_segments"]
    assert policy.predict_public(public)["probabilities"] == policy.predictor.predict(state)["probabilities"]


def test_original_prior_stays_none_and_titles_are_known_before_body(frozen):
    policy, cycles = frozen
    public = cycles[1]["public_before"]
    state, visible, _ = policy._visible(public)
    assert state["prior_probabilities"] is None and visible["before"]["prior_probabilities"] is None
    assert not state["documents"] and state["observed_links"]
    forecast = policy.predict_public(public)
    assert forecast["known_character_features"] > 0 and forecast["modeled_read_documents"] == 0
    assert forecast["probabilities"] == policy.predictor.predict(state)["probabilities"]
    without_titles = deepcopy(state)
    without_titles["observed_links"] = []
    assert policy.predictor.predict(without_titles)["fallback_reason"] == "no_known_visible_text"
    assert policy.predictor.predict(without_titles)["probabilities"] != forecast["probabilities"]


def test_hidden_outcomes_and_unread_returns_do_not_change_prediction_or_selection(frozen):
    policy, cycles = frozen
    cycle = cycles[-1]
    public = deepcopy(cycle["public_before"])
    public.update(outcome=2, score=[0, 99], future_returns={"unread": "FAKE_UNREAD_BODY"})
    public["state"]["hidden_outcome"] = 0
    public["state"]["open_text"]["unread_documents"] = [{"text": "FAKE_UNREAD_BODY", "outcome": 1}]
    baseline, changed = plan(policy, cycle), plan(policy, cycle, public)
    assert policy._visible(public)[0] == policy._visible(cycle["public_before"])[0]
    for key in ("prediction", "candidates", "selection", "physical_budget"):
        assert changed[key] == baseline[key]
    # Audit hashes correctly notice changed supplied bytes; the decision does not.
    assert changed["state_digest"] != baseline["state_digest"]
    actions = deepcopy(cycle["generated_actions"])
    actions[0]["unread_body"] = "FAKE_UNREAD_BODY"
    with pytest.raises(ValueError, match="returns, labels or arbitrary fields"):
        plan(policy, cycle, actions=actions)


def test_unobserved_seed_is_unknown_not_computed_or_positive(frozen):
    policy, cycles = frozen
    decision = plan(policy, cycles[0])
    seed = next(c for c in decision["candidates"] if c["tool"] == "read_url")
    assert seed["feasible"] and seed["value"]["status"] == "INELIGIBLE"
    assert seed["value"]["reason"] == "candidate_title_not_yet_observed"
    assert seed["value"]["expected_signed_gain"] is seed["value"]["net_voi"] is None
    assert seed["value"]["gross_evsi"] is None
    assert decision["selection"]["decision"] == "defer"


def test_same_cost_does_not_make_an_attempted_request_feasible(frozen):
    policy, cycles = frozen
    before, after = plan(policy, cycles[0]), plan(policy, cycles[1])
    seed = next(c for c in before["candidates"] if c["tool"] == "read_url")
    repeated = next(c for c in after["candidates"] if c["action_id"] == seed["action_id"])
    assert repeated["cost"] == seed["cost"]
    assert not repeated["feasible"] and repeated["reason"] == "already_attempted"
    assert repeated["value"] is None


@pytest.mark.parametrize("change", ["swap_roles", "different_match", "supplied_prior"])
def test_runtime_entry_preserves_ordered_fixture_and_training_prior(frozen, change):
    policy, cycles = frozen
    state = deepcopy(cycles[0]["public_before"]["state"])
    if change == "swap_roles":
        base = state["base"]
        base["home_team_id"], base["away_team_id"] = base["away_team_id"], base["home_team_id"]
    elif change == "different_match":
        state["match_id"] = "another-fixture"
    else:
        state["open_text"]["prior_probabilities"] = [.9, .05, .05]
    with pytest.raises(ValueError):
        policy.public_state(state)


def test_versioned_projection_and_gain_baseline_match_actual_new_run(frozen):
    folder = ROOT / 'var/submission/p3-single-match/live-03'
    if not (folder / 'run.json').exists():
        pytest.skip('Separately archived real corrected run not present')
    config = json.loads((folder / 'configuration.json').read_text(encoding='utf-8'))
    manifest = config['policy_manifest']
    model = HistoricalContextModel.load(folder / 'context-predictor.json')
    assert model.metadata['text_projection'] == 'observed-news-titles-v1'
    policy = HistoricalContextVOIPolicy(frozen[0].base_policy, model,
        fixture=manifest['fixture'], artifact=(folder / 'context-predictor.json').read_bytes(),
        state_generalization=manifest['state_generalization'])
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    final = run['cycles'][-1]['public_before']
    state, visible, _ = policy._visible(final)
    prediction = policy.predict_public(final)
    assert prediction == final['prediction']
    assert prediction['gain_model_summary']['constant_baseline_selected'] is True
    assert prediction['gain_model_summary']['conditioning_claim'] == 'not_learned_constant_gain_baseline'
    audit = prediction['text_model_input']
    assert audit['modeled_body_documents'] == audit['modeled_body_paragraphs'] == 1
    assert audit['excluded_link_labels'] > 0
    assert not {'Read', 'Previous', 'Next', 'Privacy Policy'} & {r['title'] for r in state['observed_links']}
    # Model and policy consume exactly the same projected observations. Removing
    # only admitted prose, retaining observed links/history, changes this f.
    without_body = deepcopy(state)
    without_body['documents'] = []
    assert policy.predictor.predict(without_body)['probabilities'] != prediction['probabilities']
    assert visible['news_link_projection']['excluded_labels'] == audit['excluded_link_labels']


def test_A_selected_base_only_does_not_turn_model_ignorance_into_zero_information_value(frozen):
    from datetime import datetime, timedelta
    path = ROOT / 'var/submission/p5-core-algorithm/A-selected-f-g/model.json'
    folder = ROOT / 'var/submission/p3-single-match/live-03'
    if not path.exists() or not (folder/'run.json').exists():
        pytest.skip('Separately preserved A-only selection experiment is absent')
    model = HistoricalContextModel.load(path)
    assert model.metadata['f_selection']['base_only_won'] is True
    config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    policy = HistoricalContextVOIPolicy(frozen[0].base_policy, model,
        fixture=config['policy_manifest']['fixture'], artifact=path.read_bytes(), state_generalization=True)
    cycle = json.loads((folder/'run.json').read_text(encoding='utf-8'))['cycles'][-1]
    public = deepcopy(cycle['public_before'])
    # This is a later-time, offline re-evaluation of already observed content;
    # neither original actor clock nor archived run is rewritten.
    public['decision_at'] = (datetime.fromisoformat(model.metadata['fit_execution_at'])+timedelta(seconds=1)).isoformat()
    public.pop('candidate_contexts', None)
    public['prediction'] = policy.predict_public(public)
    assert public['prediction']['probabilities'] == model.metadata['empirical_base']
    assert public['prediction']['context_signal_enabled'] is False
    assert public['prediction']['modeled_read_documents'] == 1
    decision = policy.plan_public(public, cycle['generated_actions'], public['budget'], candidate_generation_complete=True)
    eligible = [c for c in decision['candidates'] if c['feasible']]
    assert eligible and all(c['value']['status'] == 'INELIGIBLE' for c in eligible)
    assert any(c['value']['reason'] == 'predictor_text_response_not_supported' for c in eligible)
    assert decision['selection']['decision'] == 'defer'
    assert decision['selection']['stop_certificate'] is None
