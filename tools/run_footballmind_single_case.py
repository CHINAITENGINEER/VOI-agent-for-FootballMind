"""One supported match, actual public queries, one frozen empirical f/g pair.

The model chooses article acquisition from observed candidates. Source entry
URLs and budgets are experiment configuration, not a hand-authored read order.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.historical_context_policy import HistoricalContextVOIPolicy
from football_agent.learned_voi_runtime import LearnedVOIRuntime, digest, _save
from football_agent.learned_voi_web import ArchivedAndWebExecutor
from football_agent.match_analysis_report import write_analysis_report
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.evaluation.frozen_period import load_frozen_models
from football_spider.storage.raw_archive import RawArchive


def run(*, case_dir, model_path, output, request_budget=6, exploration_budget=3,
        request_loss_cost=.001, seed=20261008, state_generalization=False, projection_version=7):
    case_dir, model_path, output = map(Path, (case_dir, model_path, output))
    if output.exists():
        raise FileExistsError('Choose a new run folder; preserve previous traces')
    if not 0 <= exploration_budget <= request_budget <= 12:
        raise ValueError('A bounded single-case request budget is required')
    case = json.loads((case_dir/'case.json').read_text(encoding='utf-8'))
    private = json.loads((case_dir/'structured-environment.json').read_text(encoding='utf-8'))
    freeze_path = ROOT/'var/parallel/agent/round175/frozen-next-period.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    _, base_model = load_frozen_models(ROOT, freeze, design_path=ROOT/'docs/research/acquisition/round175-design.md')
    initial = deepcopy(case['initial_state'])
    names = {x['side']: x['canonical_name'] for x in case['source_profile']['teams']}
    fixture = {'physical_match_id': initial['match_id'], 'match_id': initial['match_id'],
        **deepcopy(initial['base']), 'home_name': names['home'], 'away_name': names['away'],
        'home_aliases': ['Arsenal'], 'away_aliases': ['Leeds', 'Leeds United']}
    identity = {'match_id': initial['match_id'], 'home_team': fixture['home_name'],
        'away_team': fixture['away_name'], 'home_aliases': fixture['home_aliases'],
        'away_aliases': fixture['away_aliases'], 'kickoff_at': fixture['kickoff'],
        'prediction_cutoff': initial['prediction_cutoff']}
    artifact = model_path.read_bytes()
    model = HistoricalContextModel.load(model_path)
    base_policy = LearnedJointVOIPolicy(base_model, lambda_cost={'requests': request_loss_cost}, loss='log_loss')
    policy = HistoricalContextVOIPolicy(base_policy, model, fixture=fixture, artifact=artifact,
        state_generalization=state_generalization)
    # P1 checked readability, not model gain. Start with the readable official
    # entry; no probed article body or headline is given to the actor for free.
    seeds = ['https://www.leedsunited.com/en?article=26856']
    hosts = ['www.leedsunited.com']
    profile = deepcopy(case['source_profile'])
    profile.update(seed_urls=seeds, allowed_hosts=hosts,
        entries=[e for e in profile['entries'] if e['url'] in seeds],
        experiment_scope='single_readable_official_club_entry_from_P1')
    generator = EvidenceDocumentCandidates(match_terms=[fixture['home_name'], fixture['away_name']],
        seed_urls=seeds, allowed_hosts=hosts, numeric_actions=[], web_cost={'money': 0., 'requests': 1.},
        match=identity, profile=profile, discovery_version=7, max_discovered_links=24,
        max_followup_queries=4)
    transport = NewsDiscoveryTools(RawArchive(output/'source-archive'), hosts,
        index_urls=seeds, projection_version=projection_version)
    executor = ArchivedAndWebExecutor(private['row'], [], transport, seed_urls=seeds)
    run_id = 'single-case-'+digest({'fixture': fixture, 'output': str(output.resolve()),
                                   'started_at': datetime.now(timezone.utc).isoformat()})[:24]
    executor.run_id = run_id
    runtime = LearnedVOIRuntime(policy, seed=seed,
        exploration_limits={'money': 0., 'requests': exploration_budget}, max_exploration_actions=exploration_budget,
        max_steps=request_budget+2)
    metadata = {'fixture': case['match'], 'generator': generator.configuration(),
        'transport': transport.configuration(), 'source_profile': profile,
        'input_bindings': {'frozen_model_config_sha256': sha256(freeze_path.read_bytes()).hexdigest(),
            'model_version': base_model.model_version, 'historical_context_version': model.model_version,
            'historical_context_sha256': sha256(artifact).hexdigest(),
            'numeric_acquisition_disabled': True, 'case_selected_by_outcome_or_query_value': False},
        'study': {'scope': 'single_match_live_acquisition_with_historical_reconstruction_training',
            'evaluation_role': 'development_case_not_independent_confirmatory_test',
            'request_cost_semantics': 'declared_loss_equivalent_per_request_not_API_price',
            'request_loss_cost': request_loss_cost, 'cost_selected_before_live_query': True,
            'state_generalization': state_generalization,
            'structured_baseline_is_separate_comparison': True, 'case_outcome_loaded': False}}
    result = runtime.run(initial, run_id=run_id, limits={'money': 0., 'requests': request_budget},
        propose=generator, execute=executor, output=output,
        provider_version=executor.version+':'+digest(transport.configuration()),
        generator_version=generator.version+':'+digest(generator.configuration()),
        run_metadata=metadata, evidence_match=identity, evidence_version='match-evidence-extraction-v5')
    report = write_analysis_report(result, match=identity, output=output)
    _save(output/'structured-comparison.json', {'forecast': case['structured_baseline'],
        'scope': case['baseline_scope'], 'used_as_text_prior': False})
    summary = {'run_id': run_id, 'output': str(output.resolve()), 'match': case['match']['match'],
        'model_version': model.model_version, 'actual_http_requests': result['measured_resources']['requests'],
        'exploration_actions': result['exploration_actions'], 'attempts': len(result['attempts']),
        'terminal_selection': result['terminal_selection'], 'analysis': report,
        'future_outcome_observed': False, 'probability_improvement_proven': False}
    _save(output/'single-case-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, default=ROOT/'var/submission/p1-single-match/initial-state')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=6)
    parser.add_argument('--exploration-requests', type=int, default=3)
    parser.add_argument('--request-loss-cost', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--projection-version', type=int, choices=[6, 7], default=7)
    parser.add_argument('--state-generalization', action='store_true',
        help='Use explicitly uncalibrated state-pattern estimates when source, candidate role and independent neighbors are supported')
    args = parser.parse_args()
    print(json.dumps(run(case_dir=args.case_dir, model_path=args.model, output=args.output,
        request_budget=args.requests, exploration_budget=args.exploration_requests,
        request_loss_cost=args.request_loss_cost, seed=args.seed,
        state_generalization=args.state_generalization,
        projection_version=args.projection_version), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
