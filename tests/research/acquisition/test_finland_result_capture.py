"""Offline result lifecycle; constructed final scores never become real labels."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket

import pytest

from football_agent import live_match_workflow as workflow
from football_research.acquisition.state import parse_time
from football_research.data_adapters import finland_result_capture as capture
from football_spider.storage.raw_archive import RawArchive
from test_finland_results import identity, league_schedule, terminal
from test_finland_fixture import identity as future_identity

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime.now(timezone.utc)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Constructed result lifecycle must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


def transport(calls, *, unavailable=False, crash=False, for_today=False):
    def request(spec, *, archive, clock):
        calls.append(spec)
        assert (archive.root/'requests'/f"{spec['stage']}.intent.json").exists()
        body = league_schedule() if spec['stage']=='schedule' else terminal()
        if for_today:
            body = body.replace(b'20261005', b'20261007').replace(b'05.10.2026', b'07.10.2026')
        obs = archive.save('open_page', spec['url'], spec['params'], b'Unavailable' if unavailable else body,
            status=503 if unavailable else 200, observed_at=clock(),
            context={'requested_at':spec['requested_at']})
        if crash:
            raise RuntimeError('Response archived but receipt completion interrupted')
        return obs
    return request


def test_due_capture_has_two_http_and_three_unchanged_cached_rules(tmp_path):
    archive, binding = identity(tmp_path)
    calls, destination = [], tmp_path/'result'
    result = capture.capture_finland_result(binding, identity_archive_root=archive.root,
        output=destination, clock=lambda:NOW, transport=transport(calls))
    assert result['status']=='VERIFIED' and len(calls)==2
    report = capture.verify_capture_receipts(binding, destination, as_of=NOW)
    assert report['http_requests_actual']==2 and len(report['cached_rule_observation_ids'])==3
    assert len(RawArchive(destination).observations())==5
    assert calls[0]['params']=={'group':'2'} and calls[1]['url']==capture.detail_url(binding)
    assert result['settlement_rows'][0]['available_at']==NOW.isoformat()
    with pytest.raises(FileExistsError):
        capture.capture_finland_result(binding, identity_archive_root=archive.root,
            output=destination, clock=lambda:NOW, transport=transport(calls))
    assert len(calls)==2


def test_not_due_copies_rules_without_request_or_result_label(tmp_path):
    archive, binding = future_identity(tmp_path)
    # A fixed pre-kickoff audit time keeps this case valid on later test dates.
    before = parse_time(binding['canonical']['kickoff'])-timedelta(hours=1)
    calls, destination = [], tmp_path/'result'
    result = capture.capture_finland_result(binding, identity_archive_root=archive.root,
        output=destination, clock=lambda:before, transport=transport(calls))
    assert result['status']=='PENDING' and result['settlement_rows']==[] and calls==[]
    assert capture.verify_capture_receipts(binding, destination, as_of=before)['http_requests_actual']==0


def test_unavailable_schedule_does_not_guess_detail_and_counts_failed_http(tmp_path):
    archive, binding = identity(tmp_path)
    calls, destination = [], tmp_path/'result'
    report = capture.capture_finland_result(binding, identity_archive_root=archive.root,
        output=destination, clock=lambda:NOW, transport=transport(calls, unavailable=True))
    assert report['settlement_rows']==[] and len(calls)==1
    assert capture.verify_capture_receipts(binding, destination, as_of=NOW)['http_requests_actual']==1
    assert not (destination/'requests/detail.intent.json').exists()


def test_interrupted_actual_request_remains_archived_without_automatic_retry(tmp_path):
    archive, binding = identity(tmp_path)
    calls, destination = [], tmp_path/'result'
    with pytest.raises(RuntimeError, match='interrupted'):
        capture.capture_finland_result(binding, identity_archive_root=archive.root,
            output=destination, clock=lambda:NOW, transport=transport(calls, crash=True))
    assert len(calls)==1 and (destination/'requests/schedule.intent.json').exists()
    assert len(RawArchive(destination).observations())==4
    assert not (destination/'capture-receipts.json').exists()
    with pytest.raises(FileExistsError):
        capture.capture_finland_result(binding, identity_archive_root=archive.root,
            output=destination, clock=lambda:NOW, transport=transport(calls))
    assert len(calls)==1


@pytest.mark.parametrize('damage',['count','cached_ids','receipt_id','intent_url','extra_raw'])
def test_receipt_reconciliation_rejects_changed_accounting(tmp_path, damage):
    archive, binding = identity(tmp_path)
    destination = tmp_path/'result'
    capture.capture_finland_result(binding, identity_archive_root=archive.root,
        output=destination, clock=lambda:NOW, transport=transport([]))
    path = destination/'capture-receipts.json'
    saved = read(path)
    if damage=='count': saved['http_requests_actual']=1
    elif damage=='cached_ids': saved['cached_rule_observation_ids']=[]
    elif damage=='receipt_id': saved['receipts'][0]['observation_id']='unknown'
    elif damage=='intent_url':
        target=destination/'requests/schedule.intent.json'
        intent=read(target); intent['url']=capture.detail_url(binding); write(target,intent)
    elif damage=='extra_raw':
        RawArchive(destination).save('open_page','https://example.test/',{},b'extra',status=200,observed_at=NOW)
    write(path,saved)
    with pytest.raises((ValueError,KeyError)):
        capture.verify_capture_receipts(binding,destination,as_of=NOW)


@pytest.mark.parametrize('unavailable,damage',[(False,False),(True,False),(False,True)])
def test_finnish_persistent_job_adopts_result_without_repeating_actor_or_http(
        tmp_path_factory,monkeypatch,unavailable,damage):
    # Copy the original pending job. All post-kickoff responses below are fake
    # transport fixtures; the original real job and its archived states stay untouched.
    original=ROOT/'var/parallel/agent/round194/live-match'
    job=tmp_path_factory.mktemp('fr')/'j'
    shutil.copytree(original,job)
    first=workflow.status_match(job)
    folder=job/first['run_directory']
    hashes=lambda: {p.relative_to(folder).as_posix():sha256(p.read_bytes()).hexdigest()
                    for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
    before=hashes()
    due=parse_time(first['result_due_at'])+timedelta(seconds=2)
    monkeypatch.setattr(workflow,'_now',lambda:due)
    import football_research.data_adapters.settlement as settlement
    class SettlementClock(datetime):
        @classmethod
        def now(cls,tz=None):
            return due.astimezone(tz) if tz is not None else due.replace(tzinfo=None)
    monkeypatch.setattr(settlement,'datetime',SettlementClock)
    calls=[]
    monkeypatch.setattr(capture,'capture_system_tls_page',transport(calls,unavailable=unavailable,for_today=True))
    adopt=workflow._adopt_settlement
    def lose_completion(*args,**kwargs):
        raise RuntimeError('interrupted after durable result receipts')
    monkeypatch.setattr(workflow,'_adopt_settlement',lose_completion)
    with pytest.raises(RuntimeError,match='durable'):
        workflow.resume_match(job,settle=True)
    count=1 if unavailable else 2
    assert len(calls)==count
    if damage:
        path=job/'results/attempt-01/capture-receipts.json'
        value=read(path); value['http_requests_actual']=1; write(path,value)
    monkeypatch.setattr(workflow,'_adopt_settlement',adopt)
    result=workflow.resume_match(job,settle=True)
    assert result['status']==('reconciliation_required' if damage else 'awaiting_result' if unavailable else 'training_eligible'), result
    assert len(calls)==count and hashes()==before and result['actor_requests']==first['actor_requests']
    if not damage:
        assert result['result_requests']==count and result['settlement_attempts']==1
        if not unavailable:
            joined=read(job/result['qualification_path'])
            assert joined['independent_result_verification'][0]['schema']=='finland-independent-regulation-settlement-v1'
            assert all(row['label']['outcome']==0 for row in joined['dataset']['episodes'])
    repeated=workflow.resume_match(job,settle=True)
    assert repeated['status']==result['status'] and len(calls)==count and hashes()==before
