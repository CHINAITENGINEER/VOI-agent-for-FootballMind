"""Package this task's evidence only; preserve all source runs and models."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'var/work/fm-live-body-chain-repair-20261025'


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',default='delivery-v2');p.add_argument('--run-id',required=True)
    args=p.parse_args()
    if not re.fullmatch(r'delivery-[A-Za-z0-9_-]+',args.delivery):raise ValueError('Plain delivery directory required')
    out=TASK/args.delivery
    assert out.is_dir() and read(out/'acceptance-report.json')['run_id']==args.run_id
    sources={
        'semantic-review':TASK/'trace/final-liverpool-semantic-review'/('run-'+args.run_id),
        'coverage':TASK/'final-coverage',
        'source-diagnosis':TASK/'final-source-diagnosis',
        'known-error-review-v1':TASK/'trace/final-liverpool-semantic-review/a807-team-news-live-review-diagnostic',
        'known-error-review-v2':TASK/'trace/final-liverpool-semantic-review/a807-team-news-live-review-v2-diagnostic',
    }
    references=[]
    for label,source in sources.items():
        if not source.is_dir():raise FileNotFoundError(str(source))
        dest=out/label;dest.mkdir(exist_ok=False)
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.csv','.txt','.md'}:
                if path.name=='manifest.json':
                    references.append({'source':str(path),'source_sha256':digest(path),'copy':None,
                        'note':'Upstream manifest references its original complete directory; use delivery root manifest for exported subset.'})
                    continue
                # Do not copy entire raw archives, code caches or model stores.
                target=dest/path.name;shutil.copyfile(path,target)
                references.append({'source':str(path),'source_sha256':digest(path),'copy':str(target.relative_to(out))})
    shutil.copyfile(TASK/'verification-tests-final.xml',out/'verification-tests-final.xml')
    catalog_path=next(p for p in sorted((TASK/'acceptance').glob('*/final-run.json'))
        if read(p)['run_id']==args.run_id).with_name('catalog-before.json')
    shutil.copyfile(catalog_path,out/'catalog-before-final-run.json')
    catalog=read(catalog_path);ready=[m for m in catalog['matches'] if m['can_start']]
    save(out/'current-coverage-summary.json',{'snapshot_at':catalog['server_time'],'catalog_updated_at':catalog['updated_at'],
        'entries':len(catalog['matches']),'startable_entries':len(ready),
        'startable_competition_labels':len({m.get('competition_key') for m in ready}),
        'scope':'source catalog records, not newly observed independent training matches or all-league body execution',
        'snapshot_path':str(catalog_path)})
    save(out/'copied-evidence-index.json',references)
    report=read(out/'acceptance-report.json')
    semantic=read(sources['semantic-review']/'semantic-review.json')
    report['independent_direct_body_semantic_review']=semantic
    save(out/'acceptance-report.json',report)
    (out/'readable-report.txt').write_text((out/'readable-report.txt').read_text(encoding='utf-8')+
        '\n最终语义逐条复核：semantic-review/readable-review.txt 与 semantic-review.json。\n'+
        'V1单一verdict复核在已知错误控制中失败，完整结果保留在known-error-review-v1；V2逐维检查/思考模式真实控制另存known-error-review-v2。\n'+
        '范围快照表为明确时点的来源条目，不是独立训练比赛数；最终正常API运行另由acceptance-report逐项给出，不混用中间运行。\n',encoding='utf-8')
    # Export only contains public football evidence, bounded receipts and hashes.
    credential_pattern=re.compile(r'(?:sk-[A-Za-z0-9]{24,}|Bearer\s+[A-Za-z0-9_.-]{24,})')
    for path in out.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.csv','.txt','.md','.xml'}:
            if credential_pattern.search(path.read_text(encoding='utf-8-sig')):
                raise RuntimeError('Credential-shaped content in export: '+str(path.relative_to(out)))
    manifest={'schema':'footballmind-live-body-delivery-manifest-v2','generated_at':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.relative_to(out)).replace('\\','/'):digest(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p!=out/'manifest.json'}}
    save(out/'manifest.json',manifest)
    assert all(digest(out/name)==value for name,value in manifest['files'].items())
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError(str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest['files'])|{'manifest.json'}
        assert all(sha256(z.read(name)).hexdigest()==value for name,value in manifest['files'].items())
    result={'delivery':str(out),'run_id':args.run_id,'manifest_files':len(manifest['files']),
        'manifest_sha256':digest(out/'manifest.json'),'zip':str(archive),'zip_sha256':digest(archive),
        'zip_bytes':archive.stat().st_size,'zip_testzip':None,'zip_matches_local':True,
        'credentials_exported':False,'source_archives_and_models_modified':False}
    save(TASK/(args.delivery+'-verification.json'),result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
