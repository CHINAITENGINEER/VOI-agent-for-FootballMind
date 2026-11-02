'''Headless match runtime. create/show/verify are offline; step/run call configured providers.'''
import argparse
import json
from pathlib import Path

from .experience import ExperienceService
from .match_proposals import MatchCandidateProvider
from .acquisition_replay import verify_run
from .value_model_adapter import CandidateValueModel
from .value_selection_policy import ValueAwareSelectionPolicy
from .runtime_voi_model import RuntimeVoIModel
from .state_bound_runtime_policy import StateBoundVoISelectionPolicy


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('var/runtime'))
    parser.add_argument('--selection-policy', choices=['baseline', 'value-aware', 'state-bound-voi'], default='baseline')
    parser.add_argument('--value-model', type=Path)
    parser.add_argument('--exploration-actions', type=int, default=0,
                        help='Opt-in finite allowance for unmeasured exact routes (1-16)')
    parser.add_argument('--exploration-cost-budget', type=float,
                        help='Total reserved declared cost in the model loss unit; not HTTP spend')
    parser.add_argument('--discovery-costs',type=Path,
                        help='Opt-in JSON costs for new search_web/read_url actions; requires exploration')
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create',help='Save match task only; no source/model calls')
    create.add_argument('--match',required=True)
    create.add_argument('--http-calls',type=int,default=8)
    create.add_argument('--llm-calls',type=int,default=5)
    create.add_argument('--seconds',type=int,default=600)
    create.add_argument('--effort',choices=['low','medium','high'],default='low')
    for name in ['show','step','run','cancel','verify','answer']:
        command=sub.add_parser(name)
        command.add_argument('run_id')
        if name=='answer':command.add_argument('text')
    args=parser.parse_args(argv)
    exploration = None
    if args.exploration_actions != 0 or args.exploration_cost_budget is not None:
        if (args.selection_policy != 'state-bound-voi' or args.exploration_actions <= 0
                or args.exploration_cost_budget is None):
            parser.error('exploration requires state-bound-voi, positive --exploration-actions and --exploration-cost-budget')
        exploration = {'max_actions': args.exploration_actions,
                       'max_declared_cost': args.exploration_cost_budget}
    discovered_routes = None
    if args.discovery_costs is not None:
        if exploration is None:
            parser.error('--discovery-costs requires explicit exploration limits')
        from .runtime_voi_model import _unique_object
        discovered_routes = json.loads(args.discovery_costs.read_text(encoding='utf-8-sig'),
                                       object_pairs_hook=_unique_object)
    if args.selection_policy == 'state-bound-voi':
        if args.value_model is None:
            parser.error('--value-model is required with --selection-policy state-bound-voi')
        value_model = RuntimeVoIModel.from_path(args.value_model)
        if value_model.evidence_basis == 'synthetic':
            parser.error('synthetic history is restricted to explicit offline test harnesses')
        selection_policy = StateBoundVoISelectionPolicy(value_model, exploration=exploration,
                                                        discovered_routes=discovered_routes)
    elif args.selection_policy == 'value-aware':
        if args.value_model is None:
            parser.error('--value-model is required with --selection-policy value-aware')
        value_model = CandidateValueModel.from_path(args.value_model)
        selection_policy = ValueAwareSelectionPolicy(
            value_model.assess, model_descriptor=value_model.configuration())
    else:
        if args.value_model is not None:
            parser.error('--value-model requires --selection-policy value-aware or state-bound-voi')
        selection_policy = None
    service=ExperienceService(args.root,proposer_factory=lambda f,e:MatchCandidateProvider(f,e),
                              selection_policy=selection_policy,
                              enable_return_assessment=True, enable_source_claims=True)
    try:
        if args.command=='create':
            output=service.create({'match':args.match,'effort':args.effort,
                'limits':{'http_calls':args.http_calls,'llm_calls':args.llm_calls,'seconds':args.seconds}})
        elif args.command=='show':output=service.get(args.run_id)
        elif args.command=='verify':output=verify_run(service.folder(args.run_id))
        elif args.command=='cancel':output=service.cancel(args.run_id)
        elif args.command=='answer':output=service.answer(args.run_id,args.text)
        elif args.command=='step':output=service.step(args.run_id)
        else:
            service._drive(args.run_id,True)
            output=service.get(args.run_id)
        print(json.dumps(output,ensure_ascii=False,indent=2))
    finally:service.close()


if __name__=='__main__':main()
