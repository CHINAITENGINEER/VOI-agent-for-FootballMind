"""Compare strict and declared state generalization on the same saved frontier.

No HTTP, result labels, replacement bodies, or downstream counterfactual
predictions are produced. This checks action eligibility, not policy benefit.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.historical_context_policy import HistoricalContextVOIPolicy
from football_agent.learned_voi_runtime import digest
from football_agent.inquiry_exploration import UniformInquiryExploration
from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.evaluation.frozen_period import load_frozen_models


def check(folder):
    folder = Path(folder)
    run = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    if run['configuration_digest'] != digest(config):
        raise ValueError('Saved configuration differs from the run')
    manifest = config['policy_manifest']
    if manifest.get('state_generalization') is not True:
        raise ValueError('Use the explicitly declared research-generalization run')
    artifact = (folder/'context-predictor.json').read_bytes()
    if sha256(artifact).hexdigest() != manifest['predictor_sha256']:
        raise ValueError('The actual frozen model changed')
    freeze_path = ROOT/'var/parallel/agent/round175/frozen-next-period.json'
    if sha256(freeze_path.read_bytes()).hexdigest() != config['supervisor_metadata']['input_bindings']['frozen_model_config_sha256']:
        raise ValueError('The underlying identity model binding changed')
    _, base = load_frozen_models(ROOT, json.loads(freeze_path.read_text(encoding='utf-8')),
        design_path=ROOT/'docs/research/acquisition/round175-design.md')
    spec = manifest['base_policy']
    base_policy = LearnedJointVOIPolicy(base, lambda_cost=spec['lambda_cost'], loss=spec['loss'],
        min_effective_matches=spec['min_effective_matches'], min_class_weight=spec['min_class_weight'])
    model = HistoricalContextModel.load(folder/'context-predictor.json')
    policies = {name: HistoricalContextVOIPolicy(base_policy, model, fixture=manifest['fixture'],
        artifact=artifact, state_generalization=flag) for name, flag in [('strict', False), ('research', True)]}
    if policies['research'].manifest != manifest:
        raise ValueError('The actual policy configuration changed')
    sampler = UniformInquiryExploration.from_configuration(config['inquiry_exploration'])
    rows, used_explorations, prior_actions = [], 0, []
    exploration_spent = {key: 0. for key in config['exploration_limits']}
    for cycle in run['cycles']:
        public = cycle['public_before']
        attempted = {a['information_key'] for a in public['attempts']}
        available = [a for a in cycle['generated_actions'] if a['information_key'] not in attempted]
        plans = {name: policy.plan_public(public, available, public['budget']) for name, policy in policies.items()}
        if plans['research'] != cycle['strict_decision']:
            raise ValueError('Current research decisions differ from the saved execution')
        summaries = {}
        for name, plan in plans.items():
            feasible = [a for a in plan['candidates'] if a['feasible']]
            computed = [a for a in feasible if a['value']['status'] == 'COMPUTED']
            summaries[name] = {'computed_candidates': len(computed),
                'nonpositive_candidate_ids': [a['action_id'] for a in computed if a['value']['net_voi'] <= 0],
                'unknown_candidate_ids': [a['action_id'] for a in feasible if a['value']['status'] != 'COMPUTED'],
                'positive_candidate_ids': [a['action_id'] for a in computed if a['value']['net_voi'] > 0]}
            eligible = [a['action_id'] for a in feasible if a['value']['status'] != 'COMPUTED'
                and used_explorations < config['max_exploration_actions']
                and all(exploration_spent[u]+a['cost'][u] <= cap+1e-12
                        for u, cap in config['exploration_limits'].items())]
            choice = plan['selection'].get('candidate_id')
            if choice is None and eligible:
                draw = sampler.select({'run_id': run['run_id'], 'configuration': config,
                    'decisions': prior_actions},
                    {'decision': {'kind': 'query', 'candidate_id': sorted(eligible)[0], 'revisit_at': None},
                     'candidates': [{'candidate_id': a['action_id']} for a in available]}, eligible)
                choice = draw['candidate_id']
            summaries[name].update(one_step_selected_id=choice,
                unknown_exploration_candidates=len(eligible),
                selected_url=next((a.get('url', a.get('query')) for a in available if a['action_id'] == choice), None))
        chosen = cycle['selected_action']
        if summaries['research']['one_step_selected_id'] != (chosen and chosen['action_id']):
            raise ValueError('The same-seed research selection must reproduce the actual action')
        rows.append({'step': cycle['step'], 'same_probabilities': plans['strict']['prediction'] == plans['research']['prediction'],
            'exploration_actions_remaining': config['max_exploration_actions']-used_explorations,
            'actual_selected_id': chosen and chosen['action_id'], 'modes': summaries})
        if cycle.get('exploration_record') is not None:
            used_explorations += 1
            exploration_spent = {u: cycle['exploration_spent'][u] for u in exploration_spent}
        if chosen is not None:
            prior_actions.append({'action_id': chosen['action_id']})
    return {'scope': 'same_public_state_same_f_g_weights_and_costs_support_rule_ablation',
        'run_id': run['run_id'], 'model_version': model.model_version,
        'network_requests': 0, 'outcome_used': False, 'counterfactual_returns_imputed': False,
        'prediction_or_resource_benefit_proven': False, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'run_id': report['run_id'], 'states_verified': len(report['rows']),
        'network_requests': 0, 'counterfactual_returns_imputed': False}, indent=2))


if __name__ == '__main__':
    main()
