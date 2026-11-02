"""Independent validation of official B gain export, using constructed raw I/O.

No real fixture label or model is created by these tests. Full 12-B/g/C
integration is tested separately by the main implementation task.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import socket
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from football_research.acquisition.optional_context_value import training_readiness
from football_research.acquisition.state import parse_time
from football_research.data_adapters import official_context_gain as gain
from football_research.data_adapters import settlement
from football_research.data_adapters.official_context_collection import collect_official_context
from football_research.data_adapters.finland_result_capture import capture_finland_result
from football_spider.storage.raw_archive import RawArchive
from test_context_gain import exposure
from test_evidence_collection import transport
from test_finland_stage_fixture import schedule
from test_official_context_collection import registered_inputs, read
from test_official_fixture_results import Source
from test_optional_prior_predictor import fit, rows


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a,**k):
        raise AssertionError('Gain validation uses constructed archived responses only')
    for owner,name in ((socket,'getaddrinfo'),(socket,'create_connection'),
                       (socket.socket,'connect'),(socket.socket,'connect_ex')):
        monkeypatch.setattr(owner,name,denied)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    root=tmp_path_factory.mktemp('gain-validation-f')
    result={}
    for domain in ('missing','provided'):
        f=fit(rows(domain),f'constructed-gain-validation-{domain}')
        path=root/(domain+'.json');f.save(path)
        result[domain]=(f,path)
    return result


def acquisition(root,model,monkeypatch):
    root.mkdir(parents=True,exist_ok=True)
    f,path=model
    registration=registered_inputs(root,path,[exposure(f)],requests=2)
    calls=transport(monkeypatch)
    summary=collect_official_context(registration,output=root/'run')
    assert summary['http_requests']==len(calls)==2
    return root/'run',calls


def audit(folder,archive,output,fpath,at=None):
    if at is None:
        return gain.build_official_context_gain_from_runs([folder],archive,output,predictor_path=fpath)
    class ConstructedAuditClock(datetime):
        @classmethod
        def now(cls,tz=None):return at
    with patch.object(gain,'datetime',ConstructedAuditClock),patch.object(settlement,'datetime',ConstructedAuditClock):
        return gain.build_official_context_gain_from_runs([folder],archive,output,predictor_path=fpath,as_of=at)


def independent_result(folder,destination):
    """Independent constructed official list + two-half final raw gateway."""
    binding=read(folder/'identity.json')
    kickoff=parse_time(binding['canonical']['kickoff'])
    at=kickoff+timedelta(hours=3)
    local=kickoff.astimezone(ZoneInfo('Europe/Helsinki'))
    class FixtureResult(Source):
        def __init__(self):
            super().__init__()
            self.native=binding['canonical']['source_event_id'].rsplit(':',1)[-1]
        def schedule(self,score='-'):
            return schedule(native=self.native,date=local.strftime('%Y%m%d'),clock=local.strftime('%H:%M'),score=score)
        def terminal(self):
            return super().terminal().replace(b'05.10.2026',local.strftime('%d.%m.%Y').encode())
    class ConstructedResultClock(datetime):
        @classmethod
        def now(cls,tz=None):return at
    with patch.object(settlement,'datetime',ConstructedResultClock):
        result=capture_finland_result(binding,identity_archive_root=folder/'identity-archive',output=destination,
            clock=lambda:at,transport=FixtureResult())
    assert result['status']=='VERIFIED' and len(result['settlement_rows'])==1
    return {binding['match_id']:destination},at


def test_pending_independent_result_keeps_costs_but_never_supplies_zero_targets(tmp_path,models,monkeypatch):
    folder,calls=acquisition(tmp_path,models['missing'],monkeypatch)
    out=tmp_path/'gains';report=audit(folder,{},out,models['missing'][1])
    assert report['supervised_transitions']==report['supervised_physical_matches']==0
    assert report['pending_prefixes']>0 and report['actual_transitions']==2
    assert report['zero_targets']==0 and read(out/'targets.json')==[]
    assert all(row['reason']=='independent_settlement_missing' for row in report['excluded_transitions'])
    assert sum(row['actual_cost']['requests'] for row in report['excluded_transitions'])==2
    ready=training_readiness(out,predictor_path=models['missing'][1],as_of=report['as_of'])
    assert not ready['ready_to_fit'] and ready['physical_matches']==0 and len(calls)==2


def test_handwritten_result_file_cannot_replace_raw_official_verification(tmp_path,models,monkeypatch):
    folder,calls=acquisition(tmp_path,models['missing'],monkeypatch)
    physical=read(folder/'identity.json')['match_id']
    fake=tmp_path/'unverified-label';fake.mkdir()
    (fake/'settlements.json').write_text(json.dumps([{'match_id':physical,'status':'final',
        'home_goals':9,'away_goals':0,'outcome':0,'available_at':datetime.now(timezone.utc).isoformat()}]),encoding='utf-8')
    report=audit(folder,{physical:fake},tmp_path/'gains',models['missing'][1])
    assert report['supervised_transitions']==0 and report['zero_targets']==0
    assert report['independent_result_verification'][0]['status']=='PENDING' and len(calls)==2


@pytest.mark.parametrize('damage',['raw','actual_forecast','model_bytes'])
def test_original_raw_actual_prediction_and_frozen_model_cannot_be_replaced(tmp_path,models,monkeypatch,damage):
    folder,calls=acquisition(tmp_path,models['missing'],monkeypatch)
    if damage=='raw':
        archive=RawArchive(folder/'source-archive');obs=archive.observations()[0]
        (archive.root/obs.payload_path).write_bytes(gzip.compress(b'replaced unseen winning article',mtime=0))
    elif damage=='actual_forecast':
        run=read(folder/'run.json');run['cycles'][0]['public_before']['prediction']['probabilities']=[.9,.05,.05]
        (folder/'run.json').write_text(json.dumps(run),encoding='utf-8')
    else:
        (folder/'context-predictor.json').write_bytes(models['provided'][1].read_bytes())
    with pytest.raises((ValueError,AssertionError)):
        audit(folder,{},tmp_path/'rejected',models['missing'][1])
    assert not (tmp_path/'rejected').exists() and len(calls)==2


def test_different_valid_f_artifact_and_future_audit_clock_are_rejected(tmp_path,models,monkeypatch):
    folder,calls=acquisition(tmp_path,models['missing'],monkeypatch)
    with pytest.raises(ValueError,match='same frozen predictor'):
        audit(folder,{},tmp_path/'wrong-f',models['provided'][1])
    with pytest.raises(ValueError,match='actual execution'):
        gain.build_official_context_gain_from_runs([folder],{},tmp_path/'future',predictor_path=models['missing'][1],
            as_of=datetime.now(timezone.utc)+timedelta(days=1))
    assert len(calls)==2


def test_independent_label_still_cannot_turn_unsupported_missing_domain_into_zero_gain(tmp_path,models,monkeypatch):
    folder,calls=acquisition(tmp_path,models['provided'],monkeypatch)
    results,at=independent_result(folder,tmp_path/'results')
    report=audit(folder,results,tmp_path/'gains',models['provided'][1],at)
    assert report['independent_result_verification'][0]['status']=='VERIFIED'
    assert report['supervised_transitions']==report['zero_targets']==0
    assert len(report['excluded_transitions'])==2
    assert all(row['reason']=='predictor_prior_domain_context_unsupported' and row['signed_gain'] is None
               for row in report['excluded_transitions'])
    assert len(calls)==2


def test_actual_supported_B_targets_are_same_f_and_compatible_with_g_without_inflating_matches(tmp_path,models,monkeypatch):
    folder,calls=acquisition(tmp_path,models['missing'],monkeypatch)
    results,at=independent_result(folder,tmp_path/'results')
    out=tmp_path/'gains';report=audit(folder,results,out,models['missing'][1],at)
    assert report['supervised_physical_matches']==1 and report['supervised_transitions']==2
    targets,features=read(out/'targets.json'),read(out/'features.json')
    assert sum(row['physical_match_weight'] for row in targets)==1
    assert all(row['inputs']['state']['prior'] is None for row in features)
    assert all(row['target_kind']=='realized_fixed_predictor_log_loss_reduction_not_expected_or_causal_value' for row in targets)
    ready=training_readiness([out,out],predictor_path=models['missing'][1],as_of=at)
    assert ready['physical_matches']==1 and ready['transitions']==2 and not ready['ready_to_fit']
    assert ready['evidence_origin']=='constructed_mechanism' and len(calls)==2
