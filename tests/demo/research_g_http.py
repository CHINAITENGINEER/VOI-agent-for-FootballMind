"""Reproduce all 5 x 3 research runs through HTTP without modifying old evidence."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
from urllib.request import Request, urlopen


def verify(url, output):
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)

    def api(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(url + path, data=data, headers={'Content-Type':'application/json'}), timeout=60) as response:
            return json.load(response)

    def save(name, value):
        (output / name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    catalog = api('/api/research-g/catalog')
    save('catalog.json',catalog)
    reference = json.loads((root/'var/work/fm-submission-g-backend-20261019-v1/evidence/verification-summary.json').read_text(encoding='utf-8'))
    results = []
    for match in catalog['matches']:
        for strategy in catalog['strategies']:
            run = api('/api/research-g/runs',{'match_id':match['match_id'],'strategy':strategy})
            save(run['run_id']+'-start.json',run)
            if not run['lifecycle']['is_terminal']:
                run = api(f'/api/research-g/runs/{run["run_id"]}/action',{'action':'run','expected_revision':run['revision']})
            report = api(f'/api/research-g/runs/{run["run_id"]}/report')
            save(run['run_id']+'-report.json',report)
            old = next(x for x in reference['runs'] if x['match_id']==run['match_id'] and x['strategy']==strategy)
            assert len(run['steps']) == old['reads']
            assert [s['action_id'] for s in run['steps']] == old['selected_actions']
            assert abs(report['evaluation']['loss_reduction']-old['evaluation']['loss_reduction']) < 1e-12
            assert run['model_bindings'] == reference['model_bindings']
            results.append({'match_id':run['match_id'],'strategy':strategy,'run_id':run['run_id'],'reads':len(run['steps']),
                            'stop_reason':run['lifecycle']['reason'],'evaluation':report['evaluation']})
    aggregate = {strategy:{'matches':5,'reads':sum(r['reads'] for r in results if r['strategy']==strategy),
                          'mean_loss_reduction':sum(r['evaluation']['loss_reduction'] for r in results if r['strategy']==strategy)/5}
                 for strategy in catalog['strategies']}
    save('verification.json',{'matched_existing_results':len(results),'runs':results,'aggregate':aggregate,'model_bindings':catalog['model_bindings']})

    # Execute the supplied reproduction script unchanged, routing only its
    # output path to our acceptance folder. The original evidence stays intact.
    original = runpy.run_path(str(root/'tools/verify_submission_research_g.py'))
    class OutputRoot:
        def __truediv__(self, child):
            if child == 'var/work/fm-submission-g-backend-20261019-v1/evidence':
                return output/'reference-script'
            return root/child
    original['main'].__globals__['ROOT'] = OutputRoot()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        original['main']()
    (output/'reference-script-output.txt').write_text(capture.getvalue(),encoding='utf-8')
    print(json.dumps({'url':url,'verified_runs':len(results),'aggregate':aggregate,'original_reproduction_script':'passed; output redirected only'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--output',type=Path,default=Path('var/work/fm-research-g-frontend-20261019-v1/http'))
    args=parser.parse_args()
    verify(args.url,args.output)
