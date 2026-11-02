"""Bounded offline audit of the raw development gateway and two label clocks.

Three existing real runs are reconstructed once. Merge fault cases reuse those
verified corpora at a test seam, avoiding repeated original-source replays.
The separate fit-ready corpus test uses explicitly constructed archived I/O.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path
import shutil
import socket

import pytest

from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.state import parse_time
from football_research.data_adapters import optional_context_development as gateway
from football_research.data_adapters.optional_context_training import build_optional_context_corpus_from_runs
from football_spider.storage.raw_archive import RawArchive
from test_prospective_training_integration import archived_training_cohort


ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT/'var/parallel/agent/round200/formal-training'
CUTOFF = '2026-10-06T20:00:00+00:00'
AS_OF = '2026-10-07T10:03:15.045268+00:00'
EARLY = 'uefa:match:2047949'
LATER_LABEL = 'uefa:match:2047948'
PENDING = 'veikkausliiga:2026:match:4353570'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')


def denied(*args, **kwargs):
    pytest.fail('Development gateway audit must not use network or fit the full corpus')


def deny_network(mp):
    for owner, key in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                       (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        mp.setattr(owner, key, denied)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    deny_network(monkeypatch)


@pytest.fixture(scope='module')
def real_sources(tmp_path_factory):
    if not (CORPUS/'report.json').exists():
        pytest.skip('The separately archived real R200 audit corpus is unavailable')
    root = tmp_path_factory.mktemp('real-development-gateway-audit')
    manifest = read(CORPUS/'report.json')
    folders = {}
    for raw in manifest['source_bindings']:
        path = Path(raw)
        if path.name != 'run.json':
            continue
        folder = path.parent
        config = read(folder/'configuration.json')
        physical = config['supervisor_metadata']['verified_fixture_identity']['canonical']['source_event_id']
        if physical in {EARLY, LATER_LABEL, PENDING}:
            folders.setdefault(physical, folder)
    assert set(folders) == {EARLY, LATER_LABEL, PENDING}
    plans = []
    for index, plan in enumerate(read(CORPUS/'registration-inputs.json')):
        path = root/f'plan-{index:03}.json'
        write(path, plan)
        plans.append(path)
    results = ROOT/'var/parallel/agent/round192/settlement/combined-result-archive'
    return {'root': root, 'folders': folders, 'plans': plans, 'results': results}


@pytest.fixture(scope='module')
def reconstructed(real_sources):
    data = real_sources
    output = data['root']/'development'
    with pytest.MonkeyPatch.context() as mp:
        deny_network(mp)
        # Even an accidental all-A fit is prohibited at this gateway boundary.
        # This small real corpus should remain below the unchanged fit gate.
        mp.setattr(OptionalPriorContextPredictor, 'fit', denied)
        report = gateway.evaluate_optional_context_from_runs(list(data['folders'].values()), data['plans'],
            data['results'], output, training_cutoff=CUTOFF, as_of=AS_OF)
    return {**data, 'output': output, 'report': report}


def test_original_real_gateways_preserve_earlier_label_time_and_exclude_late_labels(reconstructed):
    data = reconstructed
    report, output = data['report'], data['output']
    early = read(output/'training-corpus/prefix-records.json')
    early_pending = read(output/'training-corpus/pending-prefixes.json')
    late = read(output/'audit-corpus/prefix-records.json')
    final = read(output/'development-records.json')
    pending = read(output/'pending-prefixes.json')
    assert {row['physical_match_id'] for row in early} == {EARLY}
    assert {row['physical_match_id'] for row in early_pending} == {LATER_LABEL}
    assert {row['physical_match_id'] for row in late} == {EARLY, LATER_LABEL}
    assert {row['physical_match_id'] for row in pending} == {PENDING}
    assert all('outcome' not in row and 'result_available_at' not in row for row in pending+early_pending)
    original = {row['prefix_id']: row for row in early}
    later = {row['prefix_id']: row for row in late}
    for row in final:
        if row['physical_match_id'] == EARLY:
            assert row == original[row['prefix_id']]
            assert parse_time(row['result_available_at']) <= parse_time(CUTOFF)
            assert parse_time(later[row['prefix_id']]['result_available_at']) > parse_time(CUTOFF)
        else:
            assert row == later[row['prefix_id']]
            assert parse_time(row['result_available_at']) > parse_time(CUTOFF)
    evaluation = report['evaluation']
    assert evaluation['split']['training_physical_match_ids'] == [EARLY]
    exclusions = {row['physical_match_id']: row['reasons'] for row in evaluation['split']['excluded']}
    assert 'independent_settlement_not_available_at_training_cutoff' in exclusions[LATER_LABEL]
    assert 'independent_settlement_pending_for_some_or_all_prefixes' in exclusions[PENDING]
    assert evaluation['input_records'] == len(final)+len(pending)
    assert report['earlier_proven_label_prefixes'] == len(early)
    assert report['status'] == 'not_ready' and report['full_corpus_model_fitted'] is False
    assert not list(output.rglob('f.json')) and evaluation['model'] is None


def test_cutoff_snapshots_copy_original_metadata_bytes_and_all_manifest_paths_survive(reconstructed):
    data = reconstructed
    source = RawArchive(data['results'])
    for name, at in (('training-results', CUTOFF), ('audit-results', AS_OF)):
        target = RawArchive(data['output']/name)
        original = {obs.id: obs for obs in source.observations() if obs.observed_at <= parse_time(at)}
        copied = {obs.id: obs for obs in target.observations()}
        assert copied.keys() == original.keys()
        for ident, observation in copied.items():
            assert observation.model_dump() == original[ident].model_dump()
            relative = Path(observation.payload_path)
            relative_meta = relative.with_name(relative.name.removesuffix('.body.gz')+'.observation.json')
            for relative_path in (relative, relative_meta):
                assert (target.root/relative_path).read_bytes() == (source.root/relative_path).read_bytes()
            assert target.read_body(observation) == source.read_body(original[ident])
    assert len(RawArchive(data['output']/'training-results').observations()) < len(source.observations())
    assert data['report']['training_result_snapshot']['later_observations_excluded']
    for phase in ('training', 'audit'):
        corpus_report = data['report']['raw_corpus_reports'][phase]
        for path, expected in corpus_report['source_bindings'].items():
            assert Path(path).is_file() and sha256(Path(path).read_bytes()).hexdigest() == expected
        for conversion in corpus_report['conversion_reports'].values():
            for path, expected in conversion['input_bindings'].items():
                assert Path(path).is_file() and sha256(Path(path).read_bytes()).hexdigest() == expected
    def strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value
    for value in strings(read(data['output']/'report.json')):
        path = Path(value)
        if path.is_absolute():
            assert not any(part.startswith(('optional-context-development-', 'optional-context-training-'))
                           for part in path.parts)


def test_modified_original_body_cannot_pass_real_development_entry(real_sources, tmp_path, monkeypatch):
    source = real_sources['folders'][EARLY]
    copied = tmp_path/'changed-original'
    shutil.copytree(source, copied)
    body = next((copied/'source-archive/raw').rglob('*.body.gz'))
    body.write_bytes(gzip.compress(b'Constructed corruption; original observation checksum retained.', mtime=0))
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', denied)
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='checksum|replay|archive|rejected'):
        gateway.evaluate_optional_context_from_runs([copied], real_sources['plans'], real_sources['results'],
            output, training_cutoff=CUTOFF, as_of=AS_OF)
    assert not (output/'report.json').exists()


@pytest.mark.parametrize('case', ['known_label_conflict', 'known_state_conflict',
                                  'pending_state_conflict', 'known_label_vanished'])
def test_dual_reconstruction_conflicts_cannot_publish_a_report(reconstructed, tmp_path, monkeypatch, case):
    data = reconstructed
    def verified_corpus_seam(folders, plans, results, output, *, as_of):
        # Inputs came from the real gateway once above. Deliberately alter a
        # copied reconstruction to test the adapter's cross-clock consistency.
        early_phase = parse_time(as_of) == parse_time(CUTOFF)
        source = data['output']/('training-corpus' if early_phase else 'audit-corpus')
        shutil.copytree(source, output)
        records = read(Path(output)/'prefix-records.json')
        pending = read(Path(output)/'pending-prefixes.json')
        if early_phase and case == 'known_label_conflict':
            records[0]['outcome'] = (records[0]['outcome']+1)%3
        elif early_phase and case == 'known_state_conflict':
            records[0]['visible_state']['before']['fixture']['home_team_id'] = 'conflicting-team'
        elif early_phase and case == 'pending_state_conflict':
            pending[0]['visible_state']['before']['fixture']['home_team_id'] = 'conflicting-team'
        elif not early_phase and case == 'known_label_vanished':
            row = next(r for r in records if r['physical_match_id'] == EARLY)
            records.remove(row)
            row.pop('outcome')
            row.pop('result_available_at')
            pending.append(row)
        write(Path(output)/'prefix-records.json', records)
        write(Path(output)/'pending-prefixes.json', pending)
        return read(Path(output)/'report.json')
    monkeypatch.setattr(gateway, 'build_optional_context_corpus_from_runs', verified_corpus_seam)
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', denied)
    output = tmp_path/'conflict'
    with pytest.raises(ValueError, match='conflict|changed|vanished'):
        gateway.evaluate_optional_context_from_runs(list(data['folders'].values()), data['plans'], data['results'],
            output, training_cutoff=CUTOFF, as_of=AS_OF)
    assert not (output/'report.json').exists()


def test_fit_ready_constructed_raw_corpus_mode_cannot_fit_all_A(archived_training_cohort, tmp_path, monkeypatch):
    data = archived_training_cohort
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', denied)
    output = tmp_path/'raw-only'
    report = build_optional_context_corpus_from_runs(data['runs'], [data['exposure']], data['result_archive'],
        output, as_of=data['as_of'])
    assert report['readiness']['ready_to_fit'] is True
    assert report['readiness']['physical_matches'] == 12
    assert report['status'] == 'raw_corpus_only' and report['corpus_only'] is True
    assert report['model_training_performed'] is False and report['model'] is None
    assert not (output/'f.json').exists()
