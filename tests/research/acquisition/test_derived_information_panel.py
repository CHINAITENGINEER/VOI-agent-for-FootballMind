"""Actual archived parent state, constructed later bodies; zero live requests."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_research.acquisition.state import parse_time
from football_research.data_adapters.information_panel import (
    DERIVED_VERSION, _check_design, _digest, _tools, freeze_panel_design, seal_panel, load_panel,
)
from football_spider.storage.raw_archive import RawArchive

PARENT = Path("var/parallel/agent/round194/live-match/analysis/match-3085210")
FROZEN = parse_time("2026-10-07T06:00:00+00:00")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Derived panel tests must not issue requests")
    for obj, key in ((socket, "create_connection"), (socket, "getaddrinfo"), (socket.socket, "connect")):
        monkeypatch.setattr(obj, key, fail)


def make(path=None, **kwargs):
    return freeze_panel_design(PARENT, step=3, frozen_at=FROZEN, output=path, discovery_version=8, **kwargs)


def test_new_menu_uses_actual_known_links_and_new_clock_only():
    before = {p: sha256(p.read_bytes()).hexdigest() for p in PARENT.rglob('*') if p.is_file()}
    d = make()
    assert d['schema'] == DERIVED_VERSION and len(d['candidates']) == 2
    assert len(d['original_generated_menu']) == 3 and len(d['derived_menu']['generated_menu']) == 7
    assert d['public_before']['documents'] == d['actor_public_before']['documents']
    assert d['public_before']['state'] == d['actor_public_before']['state']
    assert parse_time(d['public_before']['decision_at']) == FROZEN
    assert parse_time(d['actor_public_before']['decision_at']) < FROZEN
    urls = {link['url'] for doc in d['public_before']['documents'] for link in doc['links']}
    assert all(c['url'] in urls for c in d['candidates'])
    ids = {a['action_id'] for a in d['candidates']}
    assert ids <= set(d['public_before']['candidate_contexts']['by_action_id'])
    assert not d['derived_menu']['actor_continuation'] and not d['derived_menu']['automatic_training_admission']
    _check_design(d)
    assert before == {p: sha256(p.read_bytes()).hexdigest() for p in PARENT.rglob('*') if p.is_file()}


def test_old_panel_default_preserves_original_exhausted_menu():
    d = freeze_panel_design(PARENT, step=3, frozen_at=FROZEN, output=None)
    assert d['schema'] == 'finite-information-panel-v1' and d['candidates'] == []
    assert 'derived_menu' not in d
    _check_design(d)


@pytest.mark.parametrize('version', [7, 9, True])
def test_only_explicit_supported_successor_is_allowed(version):
    with pytest.raises(ValueError, match='version 8'):
        freeze_panel_design(PARENT, step=3, frozen_at=FROZEN, output=None, discovery_version=version)


def test_cannot_pick_desired_article_or_forge_rehashed_design():
    d = make()
    with pytest.raises(ValueError, match='first two'):
        make(candidate_ids=[d['candidates'][1]['action_id'], d['candidates'][0]['action_id']])
    forged = deepcopy(d)
    forged['derived_menu']['generated_menu'][0]['cost']['money'] = 0
    forged['panel_id'] = 'panel-'+_digest({k:v for k,v in forged.items() if k!='panel_id'})[:24]
    with pytest.raises(ValueError, match='frozen actor archive'):
        _check_design(forged)


def test_new_design_cannot_backdate_or_cross_cutoff():
    for at in ('2026-10-07T05:00:00Z', '2026-10-07T16:00:00Z'):
        with pytest.raises(ValueError, match='precede prediction cutoff'):
            freeze_panel_design(PARENT, step=3, frozen_at=at, output=None, discovery_version=8)


def test_constructed_returns_stay_hidden_until_individual_reveal(tmp_path):
    from football_research.acquisition.visible_state import encode_visible_state
    path = tmp_path/'design.json'
    d = make(path)
    archive = RawArchive(tmp_path/'raw')
    captures = []
    for i, action in enumerate(d['candidates']):
        at = FROZEN+timedelta(seconds=10+i)
        tools = _tools(archive, d['transport'], lambda: at)
        request = d['prepared'][action['action_id']]
        body = f'<html><main>OnlyUnseenBody{i} content returned now.</main></html>'.encode()
        rid = f'constructed-{i}'
        observation = archive.save(request['source'], request['endpoint'], request['params'], body,
            status=200, observed_at=at, context={'requested_at':(at-timedelta(seconds=1)).isoformat(),
            'inquiry_run_id':d['panel_id'],'inquiry_request_id':rid})
        captures.append({'action_id':action['action_id'], 'decision_at':(at-timedelta(seconds=2)).isoformat(),
            'returned_at':(at+timedelta(seconds=1)).isoformat(), 'prepared':request,
            'result':{'document':tools.project(observation,request),'observation':observation.model_dump(mode='json'),
                'request_id':rid,'http_attempts':1,'latency_seconds':.1,'money':None,'money_status':'not_measured'}})
    output = tmp_path/'panel.json'
    panel = seal_panel(path,captures,panel_as_of=FROZEN+timedelta(seconds=20),archive_folder=archive.root,output=output)
    assert panel['schema'] == DERIVED_VERSION and panel['audit']['actual_http_attempts'] == 2
    env = load_panel(output)
    assert 'OnlyUnseenBody' not in json.dumps(env.public_state())
    encoded = encode_visible_state(env.public_state(), d['candidates'])
    assert encoded['before']['history']['order_matches_attempts']
    env.reveal(d['candidates'][0]['action_id'])
    assert 'OnlyUnseenBody0' in json.dumps(env.public_state())
    assert 'OnlyUnseenBody1' not in json.dumps(env.public_state())
    assert env.public_state()['panel_budget']['requests_spent'] == 1
    encoded = encode_visible_state(env.public_state(), d['candidates'])
    assert encoded['before']['history']['order_matches_attempts']
    assert len(encoded['before']['history']['attempts']) == 4
    assert encoded['before']['budget']['spent']['requests'] == 1
    assert env.public_state()['historical_actor_budget']['spent']['requests'] == 3
