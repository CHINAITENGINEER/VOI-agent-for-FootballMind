"""Small real-archive admission audit for the later research role projection.

No source request or predictor fit is permitted. Original Actor snapshots are
compared before/after and never supplied with a synthetic role or result label.
"""
from copy import deepcopy
from hashlib import sha256
import gzip
from pathlib import Path
import shutil
import socket

import pytest

from football_research.acquisition.fixture_roles import role_state, validate_role_state
from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.role_context_predictor import RoleContextPredictor
from football_research.acquisition.state import parse_time
from football_research.data_adapters.role_context_development import evaluate_role_context_from_runs
from football_research.data_adapters.role_context_training import train_role_context_from_runs
from football_spider.storage.raw_archive import RawArchive
from test_optional_context_development_gateway import (
    real_sources, read, write, denied, deny_network, ROOT, CORPUS, CUTOFF, AS_OF, EARLY, LATER_LABEL, PENDING,
)


@pytest.fixture(autouse=True)
def never_network_or_fit(monkeypatch):
    deny_network(monkeypatch)
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', denied)
    monkeypatch.setattr(RoleContextPredictor, 'fit', denied)


@pytest.fixture(scope='module')
def role_development(real_sources):
    data=real_sources
    output=data['root']/'role-development'
    with pytest.MonkeyPatch.context() as patch:
        deny_network(patch)
        patch.setattr(OptionalPriorContextPredictor,'fit',denied)
        patch.setattr(RoleContextPredictor,'fit',denied)
        report=evaluate_role_context_from_runs(list(data['folders'].values()),data['plans'],data['results'],
            output,training_cutoff=CUTOFF,as_of=AS_OF)
    return {**data,'output':output,'report':report}


def test_dual_cutoff_roles_preserve_original_visible_states_and_earlier_labels(role_development):
    data=role_development;output=data['output'];report=data['report']
    early={row['prefix_id']:row for row in read(output/'training-corpus/prefix-records.json')}
    late={row['prefix_id']:row for row in read(output/'audit-corpus/prefix-records.json')}
    pending={row['prefix_id']:row for row in read(output/'audit-corpus/pending-prefixes.json')}
    labelled=read(output/'development-records.json');unlabelled=read(output/'pending-prefixes.json')
    assert {row['physical_match_id'] for row in labelled}=={EARLY,LATER_LABEL}
    assert {row['physical_match_id'] for row in unlabelled}=={PENDING}
    for row in labelled+unlabelled:
        state=role_state(row['visible_state'])
        assert validate_role_state(state)==state and state['physical_match_id']==row['physical_match_id']
        original=early.get(row['prefix_id'],late.get(row['prefix_id'],pending.get(row['prefix_id'])))
        restored=deepcopy(row);restored['visible_state'].pop('fixture_roles')
        assert restored==original
        if row['physical_match_id']==EARLY:
            assert parse_time(row['result_available_at'])<=parse_time(CUTOFF)
            assert parse_time(late[row['prefix_id']]['result_available_at'])>parse_time(CUTOFF)
        elif row['physical_match_id']==LATER_LABEL:
            assert parse_time(row['result_available_at'])>parse_time(CUTOFF)
        else:
            assert 'outcome' not in row and 'result_available_at' not in row
    assert report['earlier_proven_label_prefixes']==len(early)
    assert report['evaluation']['split']['training_physical_match_ids']==[EARLY]
    assert report['evaluation']['model'] is None and report['full_corpus_model_fitted'] is False
    assert report['original_actor_role_features_changed'] is False and not list(output.rglob('f.json'))
    original_archive=RawArchive(data['results'])
    snapshot=RawArchive(output/'training-results')
    assert len(snapshot.observations())<len(original_archive.observations())
    for observation in snapshot.observations():
        assert observation.observed_at<=parse_time(CUTOFF)
        raw=Path(observation.payload_path)
        meta=raw.with_name(raw.name.removesuffix('.body.gz')+'.observation.json')
        for relative in (raw,meta):
            assert (snapshot.root/relative).read_bytes()==(original_archive.root/relative).read_bytes()
    for path,expected in report['input_bindings'].items():
        assert sha256(Path(path).read_bytes()).hexdigest()==expected


def test_six_actual_labels_stay_not_ready_without_calling_any_fit(real_sources,tmp_path):
    facts={row['physical_match_id'] for row in read(CORPUS/'prefix-records.json')}
    folders={}
    for raw in read(CORPUS/'report.json')['source_bindings']:
        path=Path(raw)
        if path.name=='run.json':
            config=read(path.parent/'configuration.json')
            physical=config['supervisor_metadata']['verified_fixture_identity']['canonical']['source_event_id']
            if physical in facts:
                folders.setdefault(physical,path.parent)
    assert len(folders)==6
    output=tmp_path/'six-role-A'
    report=train_role_context_from_runs(list(folders.values()),[read(p) for p in real_sources['plans']],
        real_sources['results'],output,as_of=AS_OF)
    assert report['status']=='not_ready' and report['model'] is None and not (output/'f.json').exists()
    assert report['supervised_physical_matches']==6
    assert report['readiness']['class_counts']=={'0':0,'1':2,'2':4}
    original={row['prefix_id']:row for row in read(output/'raw-corpus/prefix-records.json')}
    for row in read(output/'prefix-records.json'):
        state=role_state(row['visible_state'])
        assert state['physical_match_id']==row['physical_match_id']
        restored=deepcopy(row);restored['visible_state'].pop('fixture_roles')
        assert restored==original[row['prefix_id']]
    assert report['registered_split_counts']['B']==report['registered_split_counts']['C']==0


@pytest.mark.parametrize('kind',['raw','identity'])
def test_original_raw_or_identity_tampering_cannot_publish_role_report(real_sources,tmp_path,kind):
    copied=tmp_path/'changed';shutil.copytree(real_sources['folders'][EARLY],copied)
    if kind=='raw':
        body=next((copied/'source-archive/raw').rglob('*.body.gz'))
        body.write_bytes(gzip.compress(b'constructed corruption',mtime=0))
    else:
        config=read(copied/'configuration.json')
        config['supervisor_metadata']['verified_fixture_identity']['canonical']['home_name']='Forged identity'
        write(copied/'configuration.json',config)
    output=tmp_path/'rejected'
    with pytest.raises(ValueError):
        evaluate_role_context_from_runs([copied],real_sources['plans'],real_sources['results'],output,
            training_cutoff=CUTOFF,as_of=AS_OF)
    assert not (output/'report.json').exists() and not list(output.rglob('f.json'))


def test_label_only_entry_and_wrong_predictor_generations_are_rejected(real_sources,tmp_path):
    label_folder=tmp_path/'caller-labels';label_folder.mkdir()
    write(label_folder/'prefix-records.json',read(CORPUS/'prefix-records.json'))
    output=tmp_path/'labels-rejected'
    with pytest.raises((ValueError,FileNotFoundError)):
        train_role_context_from_runs([label_folder],[read(p) for p in real_sources['plans']],
            real_sources['results'],output,as_of=AS_OF)
    assert not (output/'report.json').exists()
    old=ROOT/'var/parallel/agent/round201/constructed-chain/f.json'
    new=ROOT/'var/parallel/agent/round203/constructed-role-chain-b12/f.json'
    assert old.is_file() and new.is_file()
    with pytest.raises(ValueError,match='schema'):RoleContextPredictor.load(old)
    with pytest.raises(ValueError,match='schema'):OptionalPriorContextPredictor.load(new)
