"""Actual frozen-platform integration through serialized public protocol only.

The trusted runner owns the environment/export and frozen model artifacts.
``decide_public`` receives only JSON public state/catalog and trained models.
This is an offline historical replay; costs describe simulated revelation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from football_platform.information import ExperimentConfig, FrozenEvidenceEnvironment
from football_platform.information.frozen import completed_bundle
from .partial import PartialPredictor
from .platform_bridge import GROUP_MAP, experiment_actions, to_research_actions, to_research_state
from .policy import select_or_stop
from .value import ValueEstimator


def decide_public(state_json: str, catalog_json: str, partial, gain, lambda_money: float, min_support=12):
    """Policy side: serialized public input only, with no environment reference."""
    if not isinstance(state_json, str) or not isinstance(catalog_json, str):
        raise TypeError('Policy boundary accepts serialized JSON only')
    state = to_research_state(json.loads(state_json))
    actions = to_research_actions(json.loads(catalog_json))
    started = time.perf_counter()
    prediction = partial.predict_partial(state)
    predict_ms = (time.perf_counter() - started) * 1000
    estimates = gain.estimate(state, actions, prediction)
    remaining = {key: max(0., limit-state['budget']['spent'].get(key, 0.))
                 for key, limit in state['budget']['limits'].items()}
    decision = select_or_stop(state, estimates, remaining, {'money': lambda_money}, min_support=min_support)
    return {'research_state': state, 'prediction': prediction, 'estimates': estimates,
            'decision': decision, 'predictor_latency_ms': predict_ms}


def _stable(output):
    result = deepcopy(output)
    result.pop('predictor_latency_ms', None)
    for row in result['estimates']:
        row.pop('estimator_batch_latency_ms', None)
        row.get('estimator_cost', {}).pop('latency_ms', None)
    return result


def _assert_public_clean(value):
    forbidden = {'home_goals', 'away_goals', 'label', 'outcome', 'input_manifest',
                 'manifest', 'snapshot_sha256', 'payload_path', 'raw_path', 'export_directory'}
    if isinstance(value, dict):
        if set(value) & forbidden:
            raise AssertionError('Private label/manifest/path entered serialized public input')
        for item in value.values():
            _assert_public_clean(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_clean(item)


def _owner_hidden_variant(manifest, records, public_state, config):
    """Synthetic owner-only counterfactual over real rows; no disk mutation."""
    changed = deepcopy(records)
    mid = public_state['match_id']
    unknown = {g for g, mask in public_state['mask'].items() if mask == 'UNACQUIRED'}
    for row in changed['matches']:
        if row['match_id'] == mid:
            row.update(home_goals=17, away_goals=19)
    for row in changed['features']:
        if row['match_id'] != mid:
            continue
        values = row['features']
        if 'strength' in unknown:
            values['elo'] = {**values['elo'], 'home': 4001., 'away': 17., 'difference': 3984.}
        for key in list(values):
            if ('rolling_xg' in unknown and '_rolling_xg_' in key) or (
                'recent_form' in unknown and '_rolling_' in key and '_rolling_xg_' not in key):
                values[key] = 99.
        if 'recent_form' in unknown:
            for side in ('home', 'away'):
                values[side+'_form']['rest_days'] = 99.
    shadow = FrozenEvidenceEnvironment(deepcopy(manifest), changed, config)
    shadow.start(mid, public_state['prediction_cutoff'], public_state['information_mode'])
    for receipt in public_state['history']:
        shadow.acquire(receipt['action_id'], receipt['request_id'])
    altered = shadow.visible_state().model_dump(mode='json')
    assert altered == public_state, 'Changing hidden values or target label changed public state'
    return altered, sorted(unknown)


def run(export, experiment, output, *, fold=2, match_id='654a90d2-65a9-5cae-8494-f2fb685a122b',
        prediction_cutoff='2025-12-29T19:30:00+00:00'):
    experiment, output = Path(experiment), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new integration report path to preserve earlier evidence')
    frozen_hashes = json.loads((experiment/'implementation.json').read_text(encoding='utf-8'))
    before = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in frozen_hashes}
    if before != frozen_hashes:
        raise ValueError('Frozen experiment implementation differs from current runtime')
    folds = json.loads((experiment/'fold-progress.json').read_text(encoding='utf-8'))
    metadata = next(row for row in folds if row['fold'] == fold)
    if match_id not in metadata['test_match_ids']:
        raise ValueError('Integration match must belong to the selected frozen test fold')
    selection = metadata['lambda_selection']['learned-net-gain/0.05']
    lambda_money = selection['selected']
    partial = PartialPredictor.load(experiment/'models'/f'fold-{fold}-partial.json')
    gain = ValueEstimator.load(experiment/'models'/f'fold-{fold}-gain.json')
    _, public_manifest = experiment_actions(seed=7)
    config = ExperimentConfig.model_validate(public_manifest['config'])
    environment = FrozenEvidenceEnvironment.from_export(export, config)
    # Full source data stays owner-side solely for counterfactual mechanism checks.
    manifest, records = completed_bundle(export)
    wire = environment.handle({'op':'start','match_id':match_id,'prediction_cutoff':prediction_cutoff})
    catalog = environment.handle({'op':'catalog'})
    assert to_research_actions(catalog) == experiment_actions(seed=7)[0]
    trajectory, invariant_checks, idempotence = [], [], []
    for step in range(4):
        _assert_public_clean(wire)
        _assert_public_clean(catalog)
        packet = decide_public(json.dumps(wire), json.dumps(catalog), partial, gain, lambda_money)
        altered, unknown = _owner_hidden_variant(manifest, records, wire, config)
        check = decide_public(json.dumps(altered), json.dumps(catalog), partial, gain, lambda_money)
        assert _stable(packet) == _stable(check)
        invariant_checks.append({'step':step,'mutated_unacquired_groups':unknown,
            'target_label_mutated':True,'public_state_equal':True,'probability_estimate_decision_equal':True,
            'classification':'synthetic_in_memory_counterfactual_over_real_frozen_episode'})
        trajectory.append({'step':step,'public_state':wire,**packet})
        decision = packet['decision']
        if decision['decision'] == 'stop':
            break
        request = {'op':'acquire','action_id':decision['action_id'],'request_id':f'research-demo-step-{step}'}
        receipt = environment.handle(request)
        next_wire = environment.handle({'op':'state'})
        duplicate = environment.handle(request)
        assert duplicate['repeated_request'] and all(part['amount']==0 for part in duplicate['charged_cost'].values())
        assert environment.handle({'op':'state'}) == next_wire
        idempotence.append({'action_id':decision['action_id'],'request_id':request['request_id'],
            'repeated_request':True,'incremental_charged_cost':duplicate['charged_cost'],
            'public_state_unchanged':True})
        trajectory[-1]['acquisition_result'] = receipt
        wire = next_wire
    else:
        raise AssertionError('Policy did not stop within the two-action experiment')
    after = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in frozen_hashes}
    assert before == after == frozen_hashes
    report = {'schema':'research-platform-environment-integration-v1','platform_environment_executed':True,
        'environment_type':'football_platform.information.FrozenEvidenceEnvironment',
        'classification':'actual_platform_environment_on_real_historical_export_with_simulated_reveal_cost',
        'fold':fold,'match_id':match_id,'prediction_cutoff':prediction_cutoff,
        'partial_version':partial.model_version,'gain_version':gain.version,'lambda_money':lambda_money,
        'lambda_selection_source':f'fold-progress.json fold {fold} learned-net-gain/0.05',
        'lambda_unit':public_manifest['lambda_money_unit'],'config':config.model_dump(mode='json'),
        'config_version':config.version,'actual_public_catalog':catalog,'trajectory':trajectory,
        'stop_reason':trajectory[-1]['decision']['reason'],
        'actions':[row['acquisition_result']['action_id'] for row in trajectory if 'acquisition_result' in row],
        'final_spent':wire['budget']['spent'],'final_mask':wire['mask'],
        'hidden_isolation_checks':invariant_checks,'idempotence_checks':idempotence,
        'frozen_implementation_hashes_unchanged':True,'implementation_sha256':before,
        'network_requests':0,'llm_tokens':0,'scope_note':'Offline cached platform execution; no real HTTP savings or single-episode policy superiority claim.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform-export',required=True)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fold',type=int,default=2)
    parser.add_argument('--match-id',default='654a90d2-65a9-5cae-8494-f2fb685a122b')
    parser.add_argument('--prediction-cutoff',default='2025-12-29T19:30:00+00:00')
    args=parser.parse_args()
    report=run(args.platform_export,args.experiment,args.output,fold=args.fold,
        match_id=args.match_id,prediction_cutoff=args.prediction_cutoff)
    print(json.dumps({key:report[key] for key in ('platform_environment_executed','actions','stop_reason','final_spent','final_mask')},ensure_ascii=False))


if __name__=='__main__':
    main()
