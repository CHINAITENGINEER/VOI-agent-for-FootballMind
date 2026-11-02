"""Fixture-scoped numerical state, independent of actor extraction (synthetic)."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import runpy

import pytest

from football_agent import runtime_source_support as support
from football_agent import runtime_source_claims as claims
from football_agent import runtime_exploration as exploration
from football_agent import runtime_discovered_routes as discovery
from football_agent.acquisition_boundary import policy_identity
from football_agent.runtime_fixture_bridge import validate_fixture_identity
from football_agent.runtime_voi_model import RuntimeVoIModel, SOURCE_SUPPORT_VERSION
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy

H = runpy.run_path(str(Path(__file__).with_name('test_runtime_source_claim_policy.py')))
URL = 'https://example.org/source-records'
CONTRACT = {'version': support.VERSION, 'sources': [{'source_id': 'test-source', 'url': URL}]}


def model_package():
    config = H['conditional_package']()
    config.update(version=SOURCE_SUPPORT_VERSION, context_semantics=support.SEMANTICS,
                  source_context_contract=deepcopy(CONTRACT))
    return config


def source_state(root, *, value='already_qualified', variant='current'):
    archive, public, originals = H['SOURCE_HELPERS']['source_public'](root)
    fixture = validate_fixture_identity(public)['candidates'][0]
    cutoff = datetime.fromisoformat(public['snapshot_as_of'])
    record = {'subject': 'home', 'predicate': 'qualification_state', 'value': value,
              'valid_from': (cutoff-timedelta(hours=1)).isoformat(),
              'valid_until': (cutoff+timedelta(hours=1)).isoformat()}
    payload = {'schema': 'football-fixture-reports-v1',
               'fixture': {'home_team': fixture['home']['name'], 'away_team': fixture['away']['name'],
                           'kickoff': fixture['kickoff'], 'competition': fixture['competition']['name']},
               'reports': [record]}
    if variant == 'previous_round':
        payload['fixture']['kickoff'] = (datetime.fromisoformat(fixture['kickoff'])-timedelta(days=7)).isoformat()
    elif variant == 'other_opponent':
        payload['fixture']['away_team'] = '另一个对手'
    elif variant == 'other_competition':
        payload['fixture']['competition'] = '另一项赛事'
    elif variant == 'expired':
        record['valid_from'] = (cutoff-timedelta(days=2)).isoformat()
        record['valid_until'] = (cutoff-timedelta(days=1)).isoformat()
    doc = public['documents'][-1]
    doc.update(url=URL, content_kind='json', excerpt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
               excerpt_truncated=False, text_truncated=False)
    if variant == 'prose':
        doc.update(content_kind='html', excerpt=fixture['home']['name']+'通常偏重防守，之前已出线。')
    public[support.CAPABILITY_FIELD] = deepcopy(CONTRACT)
    H['SOURCE_HELPERS']['rebind'](archive, public, originals)
    public[support.STATE_FIELD] = support.project_context(public)
    return public, payload


def choose(public, items=None, config=None, **options):
    policy = StateBoundVoISelectionPolicy(RuntimeVoIModel(config or model_package()),
                                        allow_synthetic=True, **options)
    public['selection_policy'] = policy_identity(policy)
    if options.get('exploration'):
        public[exploration.STATE_FIELD] = exploration.project_state(
            public['selection_policy'], [], [], public['match_input'])
    batch = H['claim_batch'](public, items or [])
    choice, diag = H['choose'](policy, batch)
    return choice, diag, batch


def test_source_record_changes_next_information_choice_without_actor_claims(tmp_path):
    qualified, _ = source_state(tmp_path/'qualified')
    must_win, _ = source_state(tmp_path/'must-win', value='must_win')
    for public, index in ((qualified, 0), (must_win, 1)):
        choice, diag, batch = choose(public)
        assert choice.candidate_id == batch.candidates[index].candidate_id
        assert diag['projection']['context']['home_reported_tactical_layout'] == 'unknown'
        assert diag['projection']['actor_semantic_assessments_used'] is False
        assert diag['source_claim_projection']['actor_assertions_used'] is False
        assert diag['source_claim_projection']['world_truth_verified'] is False
        assert diag['source_claim_projection']['numeric_voi_training_eligible'] is False
        assert diag['version'] == 'state-bound-runtime-policy-v6'
        assert diag['projection']['version'] == 'runtime-observed-context-v3'


@pytest.mark.parametrize('variant', ['previous_round', 'other_opponent', 'other_competition', 'expired', 'prose'])
def test_inapplicable_material_does_not_inherit_same_team_historical_value(tmp_path, variant):
    public, _ = source_state(tmp_path, variant=variant)
    choice, diag, _ = choose(public)
    assert choice.terminal_action == 'defer'
    assert diag['projection']['context']['home_reported_qualification_state'] == 'unknown'
    assert diag['projection']['context']['home_reported_tactical_layout'] == 'unknown'
    assert diag['source_claim_projection']['excluded_documents']


@pytest.mark.parametrize('predicate,value', [('tactical_layout','defensive'),
    ('rotation_risk','high'), ('qualification_state','must_win')])
def test_actor_relabels_qualification_quote_without_changing_numeric_context(tmp_path, predicate, value):
    public, payload = source_state(tmp_path)
    doc = public['documents'][-1]
    assertion = {'target_key': 'spftll:1001', 'subject':'home',
        'subject_name':payload['fixture']['home_team'], 'predicate':predicate, 'value':value,
        'document_id':doc['document_id'], 'quote':doc['excerpt'],
        'validity': {'from':payload['reports'][0]['valid_from'],
                     'until':payload['reports'][0]['valid_until'], 'quote':doc['excerpt']}}
    baseline, before, _ = choose(public)
    choice, after, batch = choose(public, [assertion])
    assert baseline.terminal_action is None and choice.candidate_id == batch.candidates[0].candidate_id
    assert before['projection'] == after['projection']
    assert after['source_claim_projection']['assertion_count'] == 1
    assert after['source_claim_projection']['ignored_assertions'][0]['reason'] == 'actor_note_not_numeric_source_record'
    assert after['source_claim_projection']['state']['home']['context']['rotation_risk'] == 'unknown'


def test_actor_invented_active_interval_does_not_revive_expired_record(tmp_path):
    public, payload = source_state(tmp_path, variant='expired')
    cutoff = datetime.fromisoformat(public['snapshot_as_of'])
    doc = public['documents'][-1]
    assertion = {'target_key':'spftll:1001', 'subject':'home', 'subject_name':payload['fixture']['home_team'],
        'predicate':'qualification_state', 'value':'already_qualified', 'document_id':doc['document_id'],
        'quote':doc['excerpt'], 'validity': {'from':(cutoff-timedelta(hours=1)).isoformat(),
            'until':(cutoff+timedelta(hours=1)).isoformat(), 'quote':doc['excerpt']}}
    choice, diag, _ = choose(public, [assertion])
    assert choice.terminal_action == 'defer'
    assert diag['projection']['context']['home_reported_qualification_state'] == 'unknown'


def test_unsupported_prose_keeps_bounded_exploration_available(tmp_path):
    public, _ = source_state(tmp_path, variant='prose')
    choice, diag, _ = choose(public, exploration={'max_actions':1, 'max_declared_cost':1.0},
                            discovered_routes={'search_web':{'bing':.1},'read_url':.1})
    assert choice.candidate_id is not None
    diagnostics = json.loads(choice.diagnostics_json)
    assert diagnostics['selection_basis'] == exploration.BASIS
    assert diagnostics['selected_value']['expected_value'] is None
    assert diag['projection']['context']['home_reported_qualification_state'] == 'unknown'


@pytest.mark.parametrize('field', ['context_semantics','source_context_contract'])
def test_new_model_requires_explicit_generation_semantics_and_contract(field):
    config = model_package()
    del config[field]
    with pytest.raises(ValueError, match='exact fields'):
        RuntimeVoIModel(config)


def test_old_extraction_history_cannot_be_loaded_as_strict_support_implicitly():
    config = model_package()
    config['context_semantics'] = claims.SEMANTICS
    with pytest.raises(ValueError, match='context_semantics'):
        RuntimeVoIModel(config)
    config = model_package()
    config.update(version='runtime-voi-model-v2', context_semantics=claims.SEMANTICS)
    with pytest.raises(ValueError, match='exact fields'):
        RuntimeVoIModel(config)


def test_source_contract_is_frozen_in_model_and_policy_fingerprint():
    config = model_package()
    model = RuntimeVoIModel(config)
    original = policy_identity(StateBoundVoISelectionPolicy(model))
    config['source_context_contract']['sources'][0]['url'] += '/changed'
    model.configuration()['source_context_contract']['sources'].clear()
    assert model.source_context_contract == CONTRACT
    changed = policy_identity(StateBoundVoISelectionPolicy(RuntimeVoIModel(config)))
    assert original['fingerprint'] != changed['fingerprint']
    assert support.policy_contract(original) == CONTRACT


@pytest.mark.parametrize('discovered', [False,True])
def test_v6_preserves_exploration_and_discovery_descriptors(discovered):
    options = {'exploration':{'max_actions':2, 'max_declared_cost':.3}}
    if discovered:
        options['discovered_routes'] = {'search_web':{'bing':.1},'read_url':.1}
    policy = StateBoundVoISelectionPolicy(RuntimeVoIModel(model_package()), **options)
    identity = policy_identity(policy)
    assert exploration.descriptor(identity)['max_actions'] == 2
    assert bool(discovery.descriptor(identity)) == discovered


@pytest.mark.parametrize('tamper',['capability','state','legacy_model'])
def test_selector_rejects_changed_support_boundary(tmp_path, tamper):
    public, _ = source_state(tmp_path)
    config = model_package()
    if tamper == 'capability':
        public[support.CAPABILITY_FIELD]['sources'].clear()
    elif tamper == 'state':
        public[support.STATE_FIELD]['context']['home_reported_tactical_layout'] = 'defensive'
    else:
        config = H['conditional_package']()
    with pytest.raises(ValueError, match='SOURCE_SUPPORT'):
        choose(public, config=config)


def test_strict_prompt_does_not_change_legacy_protocol():
    from football_agent.experience import ExperienceProtocol
    old = claims.extend_protocol(ExperienceProtocol)
    new = claims.extend_protocol(ExperienceProtocol, source_support=True)
    assert old.prompt == ExperienceProtocol.prompt + claims.PROMPT
    assert old.version == ExperienceProtocol.version + '-source-claims-v1'
    assert new.version.endswith('-structured-support-v1')
    assert new.schema == old.schema
    assert 'source_support_state' in new.prompt
