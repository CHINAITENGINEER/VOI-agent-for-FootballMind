"""Verify round-five model hashes, chronological isolation and reported scores."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .experiment_round4 import _summarize
from .partial import PartialPredictor
from .risk_calibrated import CalibratedRiskAwareValueEstimator
from .state import parse_time
from .value import ValueEstimator
from .verify_round4 import _rows, _walk, FORBIDDEN


def verify(path):
    path = Path(path)
    report = json.loads((path/'report.json').read_text(encoding='utf-8'))
    design = json.loads((path/'design.json').read_text(encoding='utf-8'))
    for name, expected in report['implementation_sha256'].items():
        if sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Implementation drift: {name}')
    models = 0
    test_ids = set()
    for fold in report['folds']:
        number = fold['fold']
        predictor = PartialPredictor.load(path/'models'/f'fold-{number}-predictor.json')
        risk = CalibratedRiskAwareValueEstimator.load(path/'models'/f'fold-{number}-risk.json')
        ValueEstimator.load(path/'models'/f'fold-{number}-v1-gain.json')
        models += 3
        train = set(fold['training_match_ids'])
        validation = set(fold['validation_match_ids'])
        test = set(fold['test_match_ids'])
        assert not train & validation and not train & test and not validation & test
        assert not test_ids & test
        test_ids.update(test)
        assert set(risk.manifest['training_match_ids']).issubset(train)
        assert set(risk.manifest['calibration_match_ids']).issubset(train)
        assert not set(risk.manifest['training_match_ids']) & set(risk.manifest['calibration_match_ids'])
        assert parse_time(risk.manifest['trained_as_of']) == parse_time(predictor.metadata['training_as_of'])
    episodes = list(_rows(path/'episodes.jsonl'))
    assert len(episodes) == report['episode_runs']
    assert len(test_ids) == report['test_matches']
    keys = set()
    for row in episodes:
        key = tuple(row[k] for k in ('fold','match_id','strategy','budget_money','seed'))
        assert key not in keys
        keys.add(key)
        p = row['final_probabilities']
        assert len(p) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-10
        assert row['cost']['money'] <= row['budget_money']+1e-10
        assert row['cost']['requests'] <= design['request_budget']
        assert row['hindsight_stop_regret'] >= 0
        assert row['actual_external_requests'] == row['actual_llm_tokens'] == 0
    recalculated, _, _ = _summarize(episodes, design)
    for key, expected in report['metrics'].items():
        for metric in ('log_loss','brier_score','money','requests','accuracy'):
            assert abs(recalculated[key][metric]-expected[metric]) < 1e-10
    events = actions = trajectories = 0
    trace_keys = set()
    for trace in _rows(path/'trajectories.jsonl'):
        trajectories += 1
        key = tuple(trace[k] for k in ('fold','match_id','strategy','budget_money','seed'))
        assert key in keys and key not in trace_keys
        trace_keys.add(key)
        for event in trace['events']:
            assert not FORBIDDEN.intersection(_walk(event['visible_state']))
            assert len(event['visible_state'].get('action_history',[])) == event['step']
            events += 1
            actions += int('charged_cost' in event)
    assert trace_keys == keys
    return {'status':'PASS','models_loaded':models,'episode_runs':len(episodes),
        'independent_test_matches':len(test_ids),'visible_state_checks':events,'acquisition_actions':actions,
        'temporal_partitions_disjoint':True,'metrics_recomputed':True,'implementation_hashes_match':True,
        'scope':'integrity and accounting, not confirmation of algorithm superiority',
        'artifacts':{name:sha256((path/name).read_bytes()).hexdigest() for name in
                     ('report.json','design.json','episodes.jsonl','risk-diagnostics.jsonl')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.run)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
