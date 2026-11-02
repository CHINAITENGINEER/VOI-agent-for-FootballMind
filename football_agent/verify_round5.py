"""Verify round-five fitted research models through the real frozen Agent bridge.

Uses the first scheduled fixture of each fold, before inspecting gain/outcome.
These are historical mechanism checks, never live or LLM effectiveness claims.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .acquisition import AcquisitionWorkflow
from .acquisition_evaluation import summarize_runs
from .research_bridge import make_backend


def verify(research, output, bundle):
    research, output = Path(research).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((research/'report.json').read_text(encoding='utf-8'))
    template = json.loads(Path('var/parallel/agent/round4/backend-v3-fold-2.json').read_text(encoding='utf-8'))
    episodes = [json.loads(s) for s in (research/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    runs, checks = [], []
    for fold in report['folds']:
        number = fold['fold']
        candidates = [r for r in episodes if r['fold'] == number and r['strategy'] == 'base-only' and r['budget_money'] == .05]
        first = min(candidates, key=lambda r: (r['kickoff'],r['match_id']))
        for risk_mode in ('mean', 'conservative'):
            strategy = 'risk-'+risk_mode
            lam = fold['lambda_selection'][strategy+'/0.05']['selected']
            config = {'partial_predictor': str(research/'models'/f'fold-{number}-predictor.json'),
                'value_estimator': str(research/'models'/f'fold-{number}-risk.json'),
                'value_estimator_kind': 'calibrated-risk-v2', 'lambda_cost': {'money':lam},
                'experiment': deepcopy(template['experiment']),
                'fixtures': [{k:first[k] for k in ('match_id','kickoff','prediction_cutoff')}],
                'selection_basis': 'first scheduled fixture per fold; no gain/outcome filtering',
                'lambda_validation_budget': {'money':.05,'requests':2,'selected_lambda_money':lam},
                'research_release': {'version':report['experiment_version'], 'fold':number,
                    'directory':str(research), 'evidence_grade':report['evidence_grade'],
                    'report_sha256':sha256((research/'report.json').read_bytes()).hexdigest()}}
            path = output/f'backend-fold-{number}-{risk_mode}.json'
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            backend = make_backend(mode='real', bundle_path=bundle, model_artifact=path)
            workflow = AcquisitionWorkflow(backend, output/'runs')
            run = workflow.start(first['match_id'], first['prediction_cutoff'], risk_mode=risk_mode,
                                 budget={'max_simulated_cost':.05, 'max_actions':2})
            final = workflow.run_to_completion(run['run_id'])
            expected = next(r for r in episodes if r['fold']==number and r['match_id']==first['match_id']
                            and r['strategy']==strategy and r['budget_money']==.05)
            p = [final['prediction']['probabilities'][k] for k in ('home','draw','away')]
            assert max(abs(a-b) for a,b in zip(p, expected['final_probabilities'])) < 1e-10, (number,risk_mode,final['reason'])
            assert [h['action_id'] for h in final['history']] == expected['actions'], (number,risk_mode)
            assert abs(final['budget']['simulated_cost']-expected['cost']['money']) < 1e-10
            assert final['budget']['llm_calls'] == 0
            audit = workflow.store.verify(final['run_id'])
            assert audit['status'] == 'OK'
            runs.append(final)
            checks.append({'fold':number,'risk_mode':risk_mode,'run_id':final['run_id'],
                'match_id':first['match_id'],'state':final['state'],'reason':final['reason'],
                'actions':expected['actions'],'simulated_cost':final['budget']['simulated_cost'],
                'research_probabilities_equal':True,'audit_events':audit['events_verified']})
            print(f'fold {number} {risk_mode}: {final["reason"]}', flush=True)
    acceptance = {'status':'PASS','checks':checks,'llm_provider':'NOT_CONFIGURED_OR_CALLED',
        'new_external_requests':0,'scope':'frozen historical bridge parity; not live prediction or strategy superiority'}
    (output/'acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'evaluation.json').write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2), encoding='utf-8')
    return acceptance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bundle', required=True)
    a = p.parse_args()
    result = verify(a.research,a.output,a.bundle)
    print(json.dumps({'status':result['status'],'checks':len(result['checks'])}))


if __name__ == '__main__':
    main()
