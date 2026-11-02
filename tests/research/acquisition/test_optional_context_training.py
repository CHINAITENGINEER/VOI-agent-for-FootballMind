"""Constructed raw-chain contract tests, never real samples or live HTTP."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.data_adapters.optional_context_training import train_optional_context_from_runs, main
from football_agent.learned_voi_runtime import digest
from test_evidence_collection import inputs, transport, collect, read
from test_prospective_training_integration import archived_training_cohort


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Optional-prior training tests must never perform HTTP')
    for obj, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                      (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(obj, name, blocked)


def evidence(root, monkeypatch):
    prepared, binding, plan, path = inputs(root)
    calls = transport(monkeypatch)
    collect(root, prepared, path, requests_per_match=1)
    return binding, plan, path, calls


def test_actual_v5_raw_gateway_preserves_pending_and_cli_never_fits_without_labels(tmp_path, monkeypatch):
    binding, plan, path, calls = evidence(tmp_path/'inputs', monkeypatch)
    run = tmp_path/'inputs/run'
    original = (run/'run.json').read_bytes()
    result = train_optional_context_from_runs([run], [plan], tmp_path/'empty-results', tmp_path/'output')
    assert result['status'] == 'not_ready' and result['model'] is None
    assert not result['model_training_performed'] and not (tmp_path/'output/f.json').exists()
    assert result['supervised_physical_matches'] == result['supervised_prefixes'] == 0
    assert result['pending_prefixes'] == 3 and result['pending_physical_matches'] == 1
    assert result['readiness']['physical_matches'] == 0
    assert result['domain_counts']['pending']['missing_prior']['physical_matches'] == 1
    assert result['registered_splits'] == {'A':[binding['match_id']], 'B':[], 'C':[]}
    assert all('outcome' not in r and r['visible_state']['before']['prior_probabilities'] is None
               for r in read(tmp_path/'output/pending-prefixes.json'))
    assert len(calls) == 1 and (run/'run.json').read_bytes() == original
    assert main(['--run', str(run), '--previous-plan', str(path), '--result-archive', str(tmp_path/'empty-results'),
                 '--output', str(tmp_path/'cli')]) == 0
    assert read(tmp_path/'cli/report.json')['model'] is None and len(calls) == 1


def test_both_raw_gateways_merge_by_actual_physical_identity_and_fit_only_settled_A(
        archived_training_cohort, tmp_path, monkeypatch):
    data = archived_training_cohort
    binding, plan, _, calls = evidence(tmp_path/'evidence', monkeypatch)
    result = train_optional_context_from_runs(data['runs']+[tmp_path/'evidence/run'],
        [data['exposure'], plan], data['result_archive'], tmp_path/'trained')
    assert result['input_runs'] == 14 and result['physical_matches'] == 13
    assert result['supervised_physical_matches'] == 12 and result['pending_physical_matches'] == 1
    assert result['supervised_prefixes'] == 39 and result['pending_prefixes'] == 3
    assert result['readiness']['class_counts'] == {'0':4, '1':4, '2':4}
    assert result['readiness']['prior_domains']['missing_prior']['physical_matches'] == 0
    assert result['status'] == 'fitted_research_only' and result['model_training_performed']
    assert result['production_ready'] is False and result['numeric_information_value'] is None
    model = OptionalPriorContextPredictor.load(tmp_path/'trained/f.json')
    assert binding['match_id'] not in model.metadata['training_physical_match_ids']
    assert model.metadata['empirical_base']['physical_matches'] == 12
    assert model.metadata['readiness'] == result['readiness']
    assert len(result['registered_splits']['A']) == 13 and not result['registered_splits']['B']
    assert result['later_body_projections_used'] is False and result['synthetic_mask_views'] == 0
    labelled = read(tmp_path/'trained/prefix-records.json')
    assert all(r['label_lineage'] and 'result_available_at' in r for r in labelled)
    assert len(calls) == 1 and result['network_requests'] == 0


@pytest.mark.parametrize('case', ['synthetic', 'B', 'unsupported'])
def test_nonreal_B_or_unsupported_run_cannot_be_relabelled_by_the_optional_entry(tmp_path, monkeypatch, case):
    _, plan, _, _ = evidence(tmp_path/'inputs', monkeypatch)
    folder = tmp_path/'inputs/run'
    config, run = read(folder/'configuration.json'), read(folder/'run.json')
    if case == 'synthetic':
        config['supervisor_metadata']['evidence_origin'] = 'constructed_mechanism'
        reason = 'Synthetic'
    elif case == 'B':
        config['supervisor_metadata']['context_calibration'] = {'registration_id':'cannot-convert-B-to-A'}
        reason = 'B/C'
    else:
        config['schema'] = 'unsupported-future-run'
        reason = 'Unsupported'
    run['schema'], run['configuration_digest'] = config['schema'], digest(config)
    for name, value in (('configuration.json', config), ('run.json', run)):
        (folder/name).write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match=reason):
        train_optional_context_from_runs([folder], [plan], tmp_path/'results', tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
