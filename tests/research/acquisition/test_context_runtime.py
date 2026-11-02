"""The same archived transport executes context f/B collection and exact replay.

All source bytes, outcomes and models in this module are constructed tests.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from football_agent.context_voi_policy import ContextVOIPolicy
from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.context_gain import prepare_context_calibration, build_context_gain_from_runs
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_spider.storage.raw_archive import RawArchive
from test_context_gain import frozen, exposure
from test_context_predictor import WORDS
from test_prospective_training_integration import _source_inputs, _score
from test_verified_collection import _collect


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('Constructed validation cannot use HTTP'))
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('Constructed validation cannot use DNS'))


def collected(frozen, tmp_path, monkeypatch):
    import football_agent.inquiry_tools as web_tools
    import football_agent.learned_voi_runtime as runtime
    import football_research.data_adapters.prospective_text as collector
    import test_verified_collection as fixtures
    at = datetime.now(timezone.utc)+timedelta(seconds=5)
    inputs = _source_inputs(tmp_path/'source', at=at, local_id=101, event_id=2050992)
    inputs['body'] = inputs['body'].replace(
        b'Croatia may rotate their starting lineup against Spain. Spain will assess player fitness before the match.',
        WORDS[0].encode())
    prior = exposure(frozen[0])
    prior_path = tmp_path/'prior-inventory.json'
    prior_path.write_text(json.dumps(prior), encoding='utf-8')
    registration_path = tmp_path/'B.json'
    registration = prepare_context_calibration(inputs['plan'],
        identity_bundle={'bindings': {inputs['binding']['match_id']: inputs['binding']}},
        previous_plans=[prior], predictor_path=frozen[1], output=registration_path,
        created_at=at+timedelta(milliseconds=100))
    tick = [at+timedelta(milliseconds=200)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            tick[0] += timedelta(milliseconds=10)
            return tick[0].astimezone(tz) if tz else tick[0].replace(tzinfo=None)
    with monkeypatch.context() as mp:
        for module in (web_tools, runtime, collector, fixtures):
            mp.setattr(module, 'datetime', Clock)
        folder, calls, report = _collect(inputs, mp, context_registration_path=registration_path,
            context_predictor_path=frozen[1], previous_learning_plan_paths=[prior_path])
    return folder, calls, report, inputs, registration, prior, at


def test_actual_context_collection_updates_before_next_action_and_replays(frozen, tmp_path, monkeypatch):
    folder, calls, report, inputs, *_ = collected(frozen, tmp_path, monkeypatch)
    config, run = read(folder/'configuration.json'), read(folder/'run.json')
    assert config['schema'] == 'learned-voi-open-execution-v4'
    assert len(calls) == report['http_requests'] == 2
    returned = [c for c in run['cycles'] if c['selected_action']]
    assert all(c['selection_basis'] == 'bounded_unknown_information_exploration' for c in returned)
    assert all(c['public_after']['decision_at'] == c['returned_at'] for c in returned)
    assert all(c['prediction_after'] == c['public_after']['prediction']['probabilities'] for c in returned)
    assert any(c['update']['probability_changed'] for c in returned)
    assert run['context_response_learning']['value_model_version'] is None
    assert not run['context_response_learning']['real_world_benefit_proven']
    for c in run['cycles']:
        assert 'context_calibration' not in c['public_before']
        assert c['public_before']['state']['open_text']['prior_probabilities'] == run['initial_probabilities']
    replay = replay_saved_run(folder, output=tmp_path/'replay', project_root=tmp_path)
    assert replay['identical_replay'] and replay['replay_network_requests'] == 0
    from football_research.data_adapters.prospective_learning import import_training_exposure
    with pytest.raises(ValueError, match='calibration exposure'):
        import_training_exposure([folder], created_at=datetime.now(timezone.utc)+timedelta(hours=1))


def test_context_B_has_actual_immediate_signed_targets_from_raw_gateway(frozen, tmp_path, monkeypatch):
    folder, _, _, inputs, registration, prior, at = collected(frozen, tmp_path, monkeypatch)
    import football_research.data_adapters.settlement as settlement
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            stamp = at+timedelta(hours=6)
            return stamp.astimezone(tz) if tz else stamp.replace(tzinfo=None)
    monkeypatch.setattr(settlement, 'datetime', Clock)
    archive = RawArchive(tmp_path/'results')
    _score(archive, inputs, 1)
    report = build_context_gain_from_runs([folder], archive.root, tmp_path/'g-supervision',
        predictor_path=frozen[1], registrations=[registration], previous_plans=[prior],
        as_of=at+timedelta(hours=6), evidence_origin='constructed_mechanism')
    assert report['supervised_transitions'] == 2
    assert report['supervised_physical_matches'] == 1
    assert report['exact_replay'][0]['identical_replay']
    assert all(t['outcome'] == 1 for t in read(tmp_path/'g-supervision/targets.json'))


def test_context_model_freeze_and_physical_holdout_precede_execution(frozen, tmp_path):
    at = datetime.now(timezone.utc)+timedelta(seconds=5)
    inputs = _source_inputs(tmp_path/'source', at=at, local_id=101, event_id=2050992)
    base = MarketPriorPolicy.from_manifest(inputs['plan']['selected'][0])
    with pytest.raises(ValueError, match='held-out'):
        ContextVOIPolicy(base, frozen[0], physical_match_id=frozen[0].metadata['training_physical_match_ids'][0])
    policy = ContextVOIPolicy(base, frozen[0], physical_match_id='unseen')
    with pytest.raises(ValueError, match='fitting|snapshot'):
        policy.check_start('2024-01-01T00:00:00+00:00')
