"""Persist new-runtime replay proof over original raw/doc/provider artifacts."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'var/work/fm-live-body-chain-repair-20261025/trace/runtime-replay-v1')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Independent replay output required')
    def offline(*a,**k):raise RuntimeError('Runtime archive replay forbids external network')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=offline
    path=ROOT/'tests/demo/test_live_body_runtime.py'
    spec=importlib.util.spec_from_file_location('live_archive_runtime_cases',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_hash=module.sha(module.SOURCE/'run.json')
    result=[]
    cases=[('positive',{}),('validation-repair-success',dict(fail_initial=True)),
           ('validation-repair-failed',dict(fail_initial=True,fail_repair=True)),
           ('validation-repair-budget-blocked',dict(fail_initial=True,provider_budget=1))]
    for name,kwargs in cases:
        store=args.output/name
        app,run,backend,policy=module.replay_current_runtime(store,**kwargs)
        if name in ('positive','validation-repair-success'):
            assert run['state']=='STOPPED' and len(run['steps'])==3
            assert max(abs(a-b) for a,b in zip(run['current'],backend.original['current']))<1e-12
            assert [d['action_id'] for d in run['decisions']]==[d['action_id'] for d in backend.original['decisions']]
        else:
            assert run['state']=='FAILED' and run['steps']==[] and run['used_input_ids']==[]
            assert run['current']==run['before']
        result.append(dict(case=name,state=run['state'],stop_reason=run['reason'],run_id=run['run_id'],
            run_path=str(app._folder(run['run_id'])/'run.json'),counts=run['counts'],
            before=run['before'],after=run['current'],body_pipeline=run['body_pipeline'],
            original_g_menu_checks=policy,new_runtime_g_selection_matches_original=
                [d['action_id'] for d in run['decisions']]==[d['action_id'] for d in backend.original['decisions']],
            raw_sha_checks=len(backend.raw_checks),semantic_calls=backend.semantic_calls,
            semantic_attempts=run['semantic_attempts'],test_fault_injection=kwargs,
            source_http=0,provider_http=0,training=False))
    assert module.sha(module.SOURCE/'run.json')==original_hash
    report=dict(schema='footballmind-live-runtime-archive-replay-v1',passed=True,
        original_run_path=str(module.SOURCE/'run.json'),original_run_sha256=original_hash,
        source_profile_and_menu_admission='original_archived_versions_replayed',
        g_selection='actual_frozen_g_no_policy_override',
        provider_adapter='original_typed_v3_claims_plus_explicit_concept_null_validated_by_new_live_protocol',
        protocol_scope='local_schema_compatibility_not_new_provider_prompt_validation',
        original_files_unchanged=True,cases=result,source_http=0,provider_http=0,training=False)
    (args.output/'runtime-replay-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={str(p.relative_to(args.output)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,output=str(args.output),cases=[dict(case=r['case'],counts=r['counts'],state=r['state']) for r in result],files=len(manifest)),ensure_ascii=False))


if __name__=='__main__':main()
