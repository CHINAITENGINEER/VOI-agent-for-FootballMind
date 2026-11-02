"""Constructed signed-supervision checks; no real f or B benefit is claimed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.learned_voi_web import _web_action
from football_research.acquisition.context_predictor import ContextPredictor
from football_research.acquisition.context_gain import (
    REGISTRATION, _digest, _signed_records, _query_input, _load_corpus,
    prepare_context_calibration, build_context_gain, build_context_gain_from_runs,
)
from football_research.acquisition.joint_planner import _cost
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.prospective_learning import EXPOSURE_VERSION, SCOPE
from test_context_predictor import visible, records, AS_OF, WORDS
from test_verified_collection import verified_inputs


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('No HTTP in context gain tests'))
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('No HTTP in context gain tests'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    model = ContextPredictor().fit(records(), as_of=AS_OF, dataset_version='constructed-gain-f',
                                  evidence_origin='constructed_mechanism')
    path = tmp_path_factory.mktemp('fixed-context')/'f.json'
    model.save(path)
    return model, path, sha256(path.read_bytes()).hexdigest()


def exposure(model):
    facts = {'home_id': 'H', 'away_id': 'A', 'competition': 'constructed', 'category': 'adult',
             'season': '2025', 'stage': 'league'}
    plan = {'schema': EXPOSURE_VERSION, 'purpose': 'f_train', 'inventory_scope': SCOPE,
        'legacy_exposed': True, 'artifacts': {}, 'prior_inventory': [], 'created_at': '2025-01-14T00:00:00+00:00',
        'matches': [{'match_id': mid, 'source_event_id': physical, 'fixture_facts': facts}
                    for mid, physical in model.metadata['training_aliases'].items()]}
    plan['cohort_id'] = EXPOSURE_VERSION+':'+_digest(plan)[:24]
    return plan


def registration(frozen, matches):
    model, path, digest = frozen
    at = '2027-01-19T00:00:00+00:00'
    plan = {'schema': REGISTRATION, 'purpose': 'calibration', 'created_at': at,
        'artifacts': {'text_model': {'path': str(path), 'sha256': digest, 'model_version': model.model_version,
            'training_as_of': model.metadata['training_as_of'], 'fit_execution_at': model.metadata['fit_execution_at'],
            'frozen_at': at}}, 'matches': matches, 'prior_inventory': [], 'evidence_origin': 'constructed_mechanism'}
    plan['registration_id'] = REGISTRATION+':'+_digest(plan)[:24]
    return plan


def transition(frozen, *, physical='B0', outcome=0, text=WORDS[0], status='OK'):
    mid, run = 'provider:'+physical, 'run:'+physical
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/', cost={'money': .01, 'requests': 1.})
    before = json.loads(json.dumps(visible(mid, text=None)).replace('2025-', '2027-'))
    after = json.loads(json.dumps(visible(mid, text=text)).replace('2025-', '2027-'))
    after['before']['decision_at'] = '2027-01-20T10:00:04+00:00'
    for doc in after['before']['documents']:
        doc['observed_at'], doc['available_at'] = '2027-01-20T10:00:02+00:00', '2027-01-20T10:00:03+00:00'
    before['candidates'] = {action['action_id']: {'action_id': action['action_id'],
        'information_key': action['information_key'], 'request': {'tool': 'read_url', 'url': action['url']},
        'declared_cost': _cost(action), 'cost_status': action['cost_status'], 'available': True,
        'already_attempted': False, 'observed_link_refs': []}}
    after['before']['history']['action_history'] = [{'action_id': action['action_id']}]
    after['before']['history']['attempts'] = [{'action_id': action['action_id'], 'status': status,
                                               'information_key': action['information_key']}]
    base = {'match_id': mid, 'physical_match_id': physical, 'run_id': run, 'outcome': outcome,
            'result_available_at': '2027-01-20T22:00:00+00:00'}
    prefixes = [{**base, 'prefix_id': run+':prefix:0', 'prefix_index': 0, 'visible_state': before},
                {**base, 'prefix_id': run+':prefix:1', 'prefix_index': 1, 'visible_state': after}]
    pair = {'run_id': run, 'step': 0, 'physical_match_id': physical, 'action': action,
        'before_prefix_id': prefixes[0]['prefix_id'], 'after_prefix_id': prefixes[1]['prefix_id'],
        'decision_at': '2027-01-20T10:00:01+00:00', 'observed_at': '2027-01-20T10:00:02+00:00',
        'available_at': '2027-01-20T10:00:03+00:00', 'response_status': status,
        'response_kind': 'article' if text else 'unknown', 'response_observation_eligible': True,
        'unobserved_alternative_returns_available': False, 'actual_cost': {'requests': 1, 'money': .01},
        'selection_probability': .5, 'selection_scope': 'uniform_visible_menu', 'selection_basis': 'finite_exploration'}
    plan = registration(frozen, [{'match_id': mid, 'source_event_id': physical}])
    return prefixes, [pair], {run: plan}


def derive(frozen, parts):
    return _signed_records(*parts, model=frozen[0], model_sha=frozen[2],
        as_of='2027-02-01T00:00:00+00:00', origin='constructed_mechanism')


def test_true_signed_loss_targets_keep_positive_negative_and_failure(frozen):
    import math
    positive = derive(frozen, transition(frozen, outcome=0))
    negative = derive(frozen, transition(frozen, outcome=1))
    failure = derive(frozen, transition(frozen, outcome=1, text=None, status='FAILED'))
    assert positive[1][0]['signed_gain'] > 0 and negative[1][0]['signed_gain'] < 0
    assert failure[1][0]['signed_gain'] == 0 and failure[1][0]['response_status'] == 'FAILED'
    for _, targets, exclusions in (positive, negative, failure):
        row = targets[0]
        assert not exclusions
        expected = math.log(row['prediction_after'][row['outcome']])-math.log(row['prediction_before'][row['outcome']])
        assert row['signed_gain'] == expected and row['physical_match_weight'] == 1.


def test_after_label_hidden_response_and_handed_in_gain_cannot_change_inputs(frozen):
    parts = transition(frozen)
    baseline = derive(frozen, parts)
    alternative = transition(frozen, text=WORDS[2], outcome=2)
    alternative[1][0].update(gain=1e9, future_response='SECRET_AFTER', hidden_family='SECRET_LABEL')
    changed = derive(frozen, alternative)
    assert baseline[0] == changed[0]
    rendered = json.dumps(changed[0])
    assert 'SECRET' not in rendered and 'outcome' not in rendered and 'signed_gain' not in rendered
    assert changed[1][0]['signed_gain'] != 1e9


def test_whole_physical_fixture_weight_and_identical_transition_dedup(frozen):
    first, second = transition(frozen, physical='B0'), transition(frozen, physical='B1')
    merged = (first[0]+second[0], first[1]+first[1]+second[1], {**first[2], **second[2]})
    _, targets, _ = derive(frozen, merged)
    assert len(targets) == 2 and all(t['physical_match_weight'] == .5 for t in targets)


@pytest.mark.parametrize('mutation', ['physical_train', 'first_query', 'freeze_after', 'wrong_f',
                                     'late_label', 'late_return', 'missing_history', 'unselected', 'counterfactual'])
def test_temporal_physical_actual_response_constraints(frozen, mutation):
    parts = transition(frozen, physical='official:0' if mutation == 'physical_train' else 'B0')
    prefixes, transitions, plans = parts
    plan = next(iter(plans.values()))
    if mutation == 'first_query':
        plan['created_at'] = plan['artifacts']['text_model']['frozen_at'] = transitions[0]['decision_at']
    elif mutation == 'freeze_after':
        plan['artifacts']['text_model']['frozen_at'] = '2027-01-21T00:00:00Z'
    elif mutation == 'wrong_f':
        plan['artifacts']['text_model']['sha256'] = '0'*64
    elif mutation == 'late_label':
        for row in prefixes:
            row['result_available_at'] = '2027-03-01T00:00:00Z'
    elif mutation == 'late_return':
        transitions[0]['available_at'] = '2027-01-21T10:00:00Z'
    elif mutation == 'missing_history':
        prefixes[1]['visible_state']['before']['history']['attempts'] = []
    elif mutation == 'unselected':
        prefixes[0]['visible_state']['candidates'] = {}
    elif mutation == 'counterfactual':
        transitions[0]['unobserved_alternative_returns_available'] = True
    plan['registration_id'] = REGISTRATION+':'+_digest({k: v for k, v in plan.items() if k != 'registration_id'})[:24]
    with pytest.raises(ValueError):
        derive(frozen, parts)


def test_prequery_parent_labels_are_bound_and_future_fields_are_ignored(frozen):
    state = json.loads(json.dumps(visible(text=WORDS[1], headline=True)).replace('2025-', '2027-'))
    link = state['before']['observed_links'][0]
    action = _web_action('read_url', link['target_url'], cost={'money': .01, 'requests': 1})
    state['candidates'][action['action_id']] = {'request': {'tool': 'read_url', 'url': action['url']},
        'information_key': action['information_key'], 'available': True, 'already_attempted': False,
        'declared_cost': _cost(action), 'cost_status': action['cost_status'], 'observed_link_refs': [link['link_ref']]}
    before = frozen[0].predict(state, physical_match_id='B1')['probabilities']
    inputs = _query_input(state, action, before)
    assert inputs['candidate']['observed_parents'][0]['labels'][0]['text'] == WORDS[1]
    state['candidates'][action['action_id']]['actual_response'] = 'UNREAD_SECRET'
    assert _query_input(state, action, before) == inputs
    link['target_url'] = 'https://www.uefa.com/news/different/'
    with pytest.raises(ValueError, match='parent link'):
        _query_input(state, action, before)


def test_registration_uses_actual_frozen_artifact_and_existing_identity_contract(frozen, verified_inputs, tmp_path):
    inputs = verified_inputs
    created = max(inputs['now'], datetime.fromisoformat(frozen[0].metadata['fit_execution_at']))+timedelta(seconds=1)
    options = dict(identity_bundle={'bindings': {inputs['binding']['match_id']: inputs['binding']}},
                   previous_plans=[exposure(frozen[0])], predictor_path=frozen[1], created_at=created)
    plan = prepare_context_calibration(inputs['plan'], output=tmp_path/'registration.json', **options)
    assert plan['artifacts']['text_model']['sha256'] == frozen[2]
    assert plan['artifacts']['text_model']['frozen_at'] == created.isoformat()
    assert plan['evidence_origin'] == 'constructed_mechanism'
    with pytest.raises(FileExistsError):
        prepare_context_calibration(inputs['plan'], output=tmp_path/'registration.json', **options)
    options['previous_plans'] = []
    with pytest.raises(ValueError, match='unknown_training_identity_inventory'):
        prepare_context_calibration(inputs['plan'], output=tmp_path/'wrong.json', **options)


def test_current_real_A_corpus_is_explicitly_zero_B_supervision(tmp_path):
    corpus = Path('var/parallel/agent/round194/evaluation/including-finland')
    report = build_context_gain(corpus, tmp_path/'A-diagnostic')
    assert report['status'] == 'not_ready' and report['supervised_transitions'] == 0
    assert report['registered_B_runs'] == 0 and report['input_runs'] == 12
    assert json.loads((tmp_path/'A-diagnostic/features.json').read_text()) == []
    assert json.loads((tmp_path/'A-diagnostic/targets.json').read_text()) == []
    assert not report['g_trained'] and report['network_requests'] == 0


def synthetic_corpus(frozen, root, *, wrong_purpose=False):
    """Explicitly constructed actor snapshots, not fake real raw documents."""
    root.mkdir()
    prefixes, pairs, plans = transition(frozen, outcome=1)
    pair, action = pairs[0], pairs[0]['action']
    mid, run_id = prefixes[0]['match_id'], prefixes[0]['run_id']
    plan = plans[run_id]
    plan['matches'][0]['fixture_facts'] = {'home_id': 'H', 'away_id': 'A', 'competition': 'constructed',
        'category': 'adult', 'season': '2027', 'stage': 'league'}
    plan['registration_id'] = REGISTRATION+':'+_digest({k: v for k, v in plan.items() if k != 'registration_id'})[:24]
    config = {'schema': 'constructed-actual-pair', 'run_id': run_id, 'supervisor_metadata': {
        'context_calibration': {'registration_id': plan['registration_id'], 'registration_sha256': _digest(plan)}}}
    if wrong_purpose:
        config['supervisor_metadata']['learning_cohort'] = {'purpose': 'f_train'}
    before = {'state': {'match_id': mid, 'prediction_cutoff': '2027-01-20T18:00:00+00:00',
        'base': {'kickoff': '2027-01-20T19:00:00+00:00'}, 'action_history': []},
        'decision_at': pair['decision_at'], 'documents': [], 'attempts': [],
        'prediction': {'probabilities': [.45, .3, .25], 'probability_basis': 'observed_market_snapshot'}}
    after = deepcopy(before)
    after['decision_at'] = '2027-01-20T10:00:04+00:00'
    after['state']['action_history'] = [action['action_id']]
    after['attempts'] = [{'action_id': action['action_id'], 'information_key': action['information_key'], 'status': 'OK'}]
    text = WORDS[0]
    after['documents'] = [{'document_id': 'actual-test-document', 'raw_sha256': 'a'*64,
        'url': action['url'], 'status': 'OK', 'kind': 'article', 'text': text, 'links': [],
        'observed_at': pair['observed_at'], 'available_at': pair['available_at'], 'source_published_at': None,
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article', 'segments': [
            {'text': text, 'start': 0, 'end': len(text), 'content_source': 'json_ld_article_body',
             'json_pointer': 'script[0].articleBody'}]}}]
    cycles = [{'step': 0, 'public_before': before, 'generated_actions': [action], 'selected_action': action},
              {'step': 1, 'public_before': after, 'generated_actions': [], 'selected_action': None}]
    run = {'schema': config['schema'], 'run_id': run_id, 'configuration_digest': _digest(config), 'cycles': cycles}
    raw = root/'original-run'
    raw.mkdir()
    save(raw/'run.json', run)
    save(raw/'configuration.json', config)
    for index, row in enumerate(prefixes):
        cycle = cycles[index]
        row['visible_state'] = encode_visible_state(cycle['public_before'], cycle['generated_actions'])
        row['lineage'] = {'run_sha256': _digest(run), 'public_before_sha256': _digest(cycle['public_before']),
                          'encoded_state_sha256': _digest(row['visible_state'])}
    previous = [exposure(frozen[0])]
    report = {'schema': 'raw-backed-context-prefixes-v1', 'evidence_origin': 'constructed_mechanism',
        'as_of': '2027-01-21T00:00:00+00:00', 'records_sha256': _digest(prefixes),
        'pending_sha256': _digest([]), 'transitions_sha256': _digest(pairs),
        'registration_inputs_sha256': _digest(previous),
        'input_bindings': {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in raw.iterdir()}}
    for name, value in [('conversion-report.json', report), ('prefix-records.json', prefixes),
                        ('pending-prefixes.json', []), ('actual-transitions.json', pairs),
                        ('registration-inputs.json', previous)]:
        save(root/name, value)
    return root, plan


def test_public_B_entry_writes_separate_before_only_inputs_and_negative_targets(frozen, tmp_path):
    corpus, plan = synthetic_corpus(frozen, tmp_path/'B-corpus')
    out = tmp_path/'gain'
    report = build_context_gain(corpus, out, predictor_path=frozen[1], registrations=[plan],
                                as_of='2027-02-01T00:00:00+00:00')
    assert report['status'] == 'supervision_built' and report['negative_targets'] == 1
    assert report['supervised_physical_matches'] == 1 and not report['g_trained']
    features = json.loads((out/'features.json').read_text(encoding='utf-8'))
    targets = json.loads((out/'targets.json').read_text(encoding='utf-8'))
    assert features[0]['sample_id'] == targets[0]['sample_id']
    assert 'outcome' not in json.dumps(features) and targets[0]['signed_gain'] < 0
    with pytest.raises(FileExistsError):
        build_context_gain(corpus, out, predictor_path=frozen[1], registrations=[plan])


def test_registered_A_run_cannot_be_relabelled_by_a_context_B_sidecar(frozen, tmp_path):
    corpus, plan = synthetic_corpus(frozen, tmp_path/'wrong-purpose', wrong_purpose=True)
    with pytest.raises(ValueError, match='original A/confirmation'):
        build_context_gain(corpus, tmp_path/'forbidden', predictor_path=frozen[1], registrations=[plan],
                           as_of='2027-02-01T00:00:00+00:00')


def test_bound_original_bytes_and_resigned_derived_hidden_state_are_rejected(frozen, tmp_path):
    corpus, _ = synthetic_corpus(frozen, tmp_path/'bound')
    rows = json.loads((corpus/'prefix-records.json').read_text())
    rows[0]['visible_state']['before']['contents'].append({'content_id': 'text-forged', 'text': 'UNREAD_FUTURE'})
    report = json.loads((corpus/'conversion-report.json').read_text())
    report['records_sha256'] = _digest(rows)
    save(corpus/'prefix-records.json', rows)
    save(corpus/'conversion-report.json', report)
    with pytest.raises(ValueError, match='unchanged original'):
        _load_corpus(corpus)
    (corpus/'original-run/run.json').write_text('{}')
    with pytest.raises(ValueError, match='input bytes changed'):
        _load_corpus(corpus)


def test_real_independent_settlement_cannot_be_replaced_even_with_new_derived_digest(tmp_path):
    source = Path('var/parallel/agent/round194/evaluation/including-finland')
    copied = tmp_path/'real-derived'
    copied.mkdir()
    names = ('conversion-report.json', 'prefix-records.json', 'pending-prefixes.json',
             'actual-transitions.json', 'registration-inputs.json')
    for name in names:
        (copied/name).write_bytes((source/name).read_bytes())
    rows = json.loads((copied/'prefix-records.json').read_text(encoding='utf-8'))
    rows[0]['outcome'] = (rows[0]['outcome']+1) % 3
    report = json.loads((copied/'conversion-report.json').read_text(encoding='utf-8'))
    report['records_sha256'] = _digest(rows)
    save(copied/'prefix-records.json', rows)
    save(copied/'conversion-report.json', report)
    with pytest.raises(ValueError, match='independent raw gateway'):
        _load_corpus(copied)


def test_public_B_raw_gateway_uses_original_prequery_registration_and_independent_result(frozen, tmp_path, monkeypatch):
    import football_agent.inquiry_tools as web_tools
    import football_agent.learned_voi_runtime as runtime
    import football_research.data_adapters.prospective_text as collector
    import test_verified_collection as fixture_module
    from test_prospective_training_integration import _source_inputs, _score
    from football_spider.storage.raw_archive import RawArchive
    from test_verified_collection import _collect
    at = datetime.now(timezone.utc)+timedelta(seconds=5)
    inputs = _source_inputs(tmp_path/'source', at=at, local_id=101, event_id=2050991)
    prior = exposure(frozen[0])
    plan = prepare_context_calibration(inputs['plan'], identity_bundle={'bindings': {
        inputs['binding']['match_id']: inputs['binding']}}, previous_plans=[prior], predictor_path=frozen[1],
        output=tmp_path/'registered-before-query.json', created_at=at+timedelta(milliseconds=100))
    clock = [at+timedelta(milliseconds=200)]
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            clock[0] += timedelta(milliseconds=10)
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)
    original_run = runtime.LearnedVOIRuntime.run
    def registered_run(self, *args, **kwargs):
        kwargs['run_metadata'] = deepcopy(kwargs['run_metadata'])
        kwargs['run_metadata']['context_calibration'] = {
            'registration_id': plan['registration_id'], 'registration_sha256': _digest(plan)}
        return original_run(self, *args, **kwargs)
    with monkeypatch.context() as mp:
        for module in (web_tools, runtime, collector, fixture_module):
            mp.setattr(module, 'datetime', FixtureClock)
        mp.setattr(runtime.LearnedVOIRuntime, 'run', registered_run)
        folder, calls, _ = _collect(inputs, mp)
    assert len(calls) == 2
    original = (folder/'run.json').read_bytes()
    results = RawArchive(tmp_path/'independent-results')
    _score(results, inputs, 1)
    # This is an explicitly simulated future B fixture. The settlement adapter
    # additionally checks actual wall time, so advance that test clock too.
    import football_research.data_adapters.settlement as settlement
    class SettlementClock(datetime):
        @classmethod
        def now(cls, tz=None):
            stamp = at+timedelta(hours=6)
            return stamp.astimezone(tz) if tz is not None else stamp.replace(tzinfo=None)
    monkeypatch.setattr(settlement, 'datetime', SettlementClock)
    output = tmp_path/'actual-B-supervision'
    report = build_context_gain_from_runs([folder], results.root, output, predictor_path=frozen[1],
        registrations=[plan], previous_plans=[prior], as_of=at+timedelta(hours=6),
        evidence_origin='constructed_mechanism')
    assert report['status'] == 'supervision_built' and report['supervised_transitions'] == 2
    assert report['registered_B_runs'] == report['supervised_physical_matches'] == 1
    targets = json.loads((output/'targets.json').read_text(encoding='utf-8'))
    assert sum(row['physical_match_weight'] for row in targets) == 1
    assert all(row['outcome'] == 1 for row in targets)
    assert (folder/'run.json').read_bytes() == original
    assert report['network_requests'] == 0 and not report['g_trained']
    assert len(report['exact_replay']) == 1 and report['exact_replay'][0]['identical_replay']
    assert report['source_inputs_file_sha256'] == sha256((output/'source-inputs.json').read_bytes()).hexdigest()
    source = json.loads((output/'source-inputs.json').read_text(encoding='utf-8'))
    assert source['input_bindings'] and source['previous_plans'] == [prior]
    with pytest.raises(ValueError, match='different frozen predictor or origin'):
        build_context_gain_from_runs([folder], results.root, tmp_path/'synthetic-cannot-be-real',
            predictor_path=frozen[1], registrations=[plan], previous_plans=[prior],
            as_of=at+timedelta(hours=6))
    # Re-signing a terminal public snapshot cannot manufacture an unread body.
    import shutil
    tampered = tmp_path/'forged-terminal'
    shutil.copytree(folder, tampered)
    fake_run = json.loads(original)
    terminal = fake_run['cycles'][-1]
    fake_document = deepcopy(next(doc for cycle in fake_run['cycles']
        for doc in cycle['public_before']['documents'] if doc.get('structured_news', {}).get('segments')))
    forged_text = WORDS[1]+' FORGED_UNREAD_TERMINAL'
    fake_document.update(document_id='never-requested-document', text=forged_text)
    fake_document['structured_news']['segments'] = [{'text': forged_text, 'start': 0, 'end': len(forged_text),
        'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]
    terminal['public_before']['documents'].append(fake_document)
    if 'public_digest' in terminal:
        terminal['public_digest'] = _digest(terminal['public_before'])
    save(tampered/'run.json', fake_run)
    save(tampered/f"cycle-{terminal['step']:03}.json", terminal)
    with pytest.raises(ValueError, match='exact replay'):
        build_context_gain_from_runs([tampered], results.root, tmp_path/'no-forged-gain',
            predictor_path=frozen[1], registrations=[plan], previous_plans=[prior],
            as_of=at+timedelta(hours=6), evidence_origin='constructed_mechanism')
    # A missing earlier raw receipt cannot leave a shorter later gain example.
    (folder/'receipt-000.json').rename(folder/'receipt-000.unavailable')
    with pytest.raises(ValueError, match='gateway|history'):
        build_context_gain_from_runs([folder], results.root, tmp_path/'missing-actual-history',
            predictor_path=frozen[1], registrations=[plan], previous_plans=[prior],
            as_of=at+timedelta(hours=6), evidence_origin='constructed_mechanism')
