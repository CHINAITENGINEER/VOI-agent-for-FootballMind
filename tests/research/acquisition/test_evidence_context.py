"""Missing-prior learning inputs retain the actual actor's null predictions."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket
from zoneinfo import ZoneInfo

import pytest

from football_research.data_adapters.evidence_context import build_evidence_context_from_runs
from test_evidence_collection import inputs, transport, collect, read
from test_evidence_live_workflow import job


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):
        pytest.fail('Evidence context conversion may not perform HTTP')
    for owner,name in ((socket,'getaddrinfo'),(socket,'create_connection'),(socket.socket,'connect')):
        monkeypatch.setattr(owner,name,denied)


def acquired(root, monkeypatch, *, fail=False):
    prepared,binding,plan,path = inputs(root)
    calls=transport(monkeypatch,fail=fail)
    collect(root,prepared,path,requests_per_match=2)
    return binding,plan,calls


def test_actual_before_and_immediate_after_are_pending_without_imputed_probabilities(tmp_path,monkeypatch):
    binding,plan,calls=acquired(tmp_path,monkeypatch)
    original=(tmp_path/'run/run.json').read_bytes()
    result=build_evidence_context_from_runs([tmp_path/'run'],[plan],tmp_path/'corpus')
    assert result['input_runs']==result['physical_matches']==1
    assert result['actual_transitions']==len(calls)==2
    assert result['pending_prefixes']==5 and result['supervised_prefixes']==0
    records=read(tmp_path/'corpus/pending-prefixes.json')
    assert all(r['visible_state']['before']['prior_probabilities'] is None for r in records)
    assert all(r['visible_state']['before']['current_probabilities'] is None for r in records)
    assert all('outcome' not in r for r in records)
    assert {r['physical_match_id'] for r in records}=={binding['match_id']}
    pairs=read(tmp_path/'corpus/actual-transitions.json')
    byid={r['prefix_id']:r for r in records}
    for pair in pairs:
        assert byid[pair['after_prefix_id']]['visible_state']['before']['decision_at']==pair['returned_at']
        assert pair['original_gain_target'] is None
    assert original==(tmp_path/'run/run.json').read_bytes()
    assert result['replay_acceptance'][0]['identical_replay']
    assert result['network_requests']==0 and not result['later_body_projections_used']
    with pytest.raises(FileExistsError):
        build_evidence_context_from_runs([tmp_path/'run'],[plan],tmp_path/'corpus')


def test_failures_are_retained_in_history_and_do_not_become_zero_gain_labels(tmp_path,monkeypatch):
    _,plan,_=acquired(tmp_path,monkeypatch,fail=True)
    result=build_evidence_context_from_runs([tmp_path/'run'],[plan],tmp_path/'corpus')
    assert result['response_statuses']=={'FAILED':2}
    records=read(tmp_path/'corpus/pending-prefixes.json')
    terminal=records[-1]['visible_state']['before']
    assert [r['status'] for r in terminal['history']['attempts']]==['FAILED','FAILED']
    assert not terminal['contents'] and not result['numeric_gain_supervision_created']


def test_missing_registration_cannot_reclassify_original_exposure(tmp_path,monkeypatch):
    acquired(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='registered in A'):
        build_evidence_context_from_runs([tmp_path/'run'],[],tmp_path/'corpus')
    assert not (tmp_path/'corpus').exists()


def test_changed_raw_actor_snapshot_is_rejected_before_a_corpus_is_written(tmp_path,monkeypatch):
    _,plan,_=acquired(tmp_path,monkeypatch)
    run=read(tmp_path/'run/run.json')
    run['cycles'][0]['public_before']['documents'].append({'text':'retroactive information'})
    (tmp_path/'run/run.json').write_text(json.dumps(run),encoding='utf-8')
    with pytest.raises(ValueError,match='Per-cycle'):
        build_evidence_context_from_runs([tmp_path/'run'],[plan],tmp_path/'corpus')
    assert not (tmp_path/'corpus').exists()


def test_repeated_input_path_does_not_increase_the_physical_sample_count(tmp_path,monkeypatch):
    _,plan,_=acquired(tmp_path,monkeypatch)
    result=build_evidence_context_from_runs([tmp_path/'run',tmp_path/'run'],[plan],tmp_path/'corpus')
    assert result['input_runs']==result['physical_matches']==1 and result['pending_prefixes']==5


def test_verified_independent_result_can_label_null_prior_prefixes_without_changing_actor(job,monkeypatch):
    from football_agent import live_match_workflow as w, evidence_live_workflow as e
    from football_research.acquisition.state import parse_time
    from football_research.data_adapters.finland_result_capture import capture_finland_result
    from football_research.data_adapters import settlement
    from test_official_fixture_results import Source
    from test_finland_stage_fixture import schedule
    started=w.start_match(job['query'],job['output'],settings_path=job['settings'])
    folder=job['output']/started['run_directory']
    original=(folder/'run.json').read_bytes()
    due=parse_time(started['result_due_at'])
    monkeypatch.setattr(w,'_now',lambda:due)
    class FutureClock(datetime):
        @classmethod
        def now(cls,tz=None):return due.astimezone(tz) if tz else due.replace(tzinfo=None)
    monkeypatch.setattr(settlement,'datetime',FutureClock)
    source=Source();source.native='4353553'
    local=job['kickoff'].astimezone(ZoneInfo('Europe/Helsinki'))
    source.schedule=lambda score='-':schedule(date=local.strftime('%Y%m%d'),clock=local.strftime('%H:%M'),score=score)
    old=source.terminal
    source.terminal=lambda:old().replace(b'05.10.2026',local.strftime('%d.%m.%Y').encode())
    monkeypatch.setattr(e,'capture_finland_result',lambda binding,**kwargs:capture_finland_result(binding,transport=source,**kwargs))
    finished=w.resume_match(job['output'],settle=True)
    assert finished['independent_labels']==1
    plan=read(job['output']/'learning-plan.json')
    result=build_evidence_context_from_runs([folder],[plan],job['root']/'corpus',as_of=due,
        result_archives={finished['source_event_id']:job['output']/finished['settlement_path']})
    assert result['supervised_prefixes']==5 and result['pending_prefixes']==0
    rows=read(job['root']/'corpus/prefix-records.json')
    assert all(r['outcome']==0 and r['visible_state']['before']['prior_probabilities'] is None for r in rows)
    assert result['legacy_market_model_training_eligible'] is False
    assert not result['numeric_gain_supervision_created']
    assert (folder/'run.json').read_bytes()==original
