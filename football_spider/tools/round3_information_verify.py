"""Real frozen episodes + existing research/Agent public consumers; zero HTTP."""
from pathlib import Path
import json
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from football_platform.information import FrozenEvidenceEnvironment, ExperimentConfig, simulated_cost
from football_spider.storage.raw_archive import atomic_write, json_bytes, digest

OUT = ROOT / 'var/parallel/collectors/round3/environment'
BUNDLE = ROOT / 'var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda'


def main():
    # Public transformations owned by the consumers; no model code is changed.
    from football_research.acquisition.platform_bridge import to_research_state, to_research_actions
    from football_agent.research_bridge import platform_state
    config = ExperimentConfig(mode='simulated_reveal',
        bindings={'results_view': ['strength', 'recent_form'], 'xg_view': ['rolling_xg']},
        costs={'results_view': simulated_cost(money=.02, latency_ms=0),
               'xg_view': simulated_cost(money=.03, latency_ms=0)},
        budget={'money': .05, 'requests': 2}, seed=7)
    env = FrozenEvidenceEnvironment.from_export(BUNDLE, config)
    fixtures = env.list_fixtures()
    results = Counter()
    mask_counts = Counter()
    public_episodes = []
    for i, fixture in enumerate(fixtures):
        trace = [env.start(fixture['match_id'], fixture['prediction_cutoff']).model_dump(mode='json')]
        assert trace[0]['groups'] == {}
        for aid in config.bindings:
            receipt = env.acquire(aid, 'batch-' + aid)
            results[receipt.status] += 1
            state = env.visible_state()
            assert all(g.times.feature_available_at <= state.prediction_cutoff for g in state.groups.values())
            research = to_research_state(state)
            agent = platform_state(state)
            assert set(research['mask']) == {k for k, v in agent['mask'].items() if v}
            trace += [receipt.model_dump(mode='json'), state.model_dump(mode='json')]
        mask_counts.update(env.visible_state().mask.values())
        if i in (200, 400, 600):
            original = env.visible_state()
            assert env.restore(original) == original
            # Same episode+seed+action IDs yields byte-identical public trace.
            repeated = [env.start(fixture['match_id'], fixture['prediction_cutoff']).model_dump(mode='json')]
            for aid in config.bindings:
                repeated += [env.acquire(aid, 'batch-' + aid).model_dump(mode='json'), env.visible_state().model_dump(mode='json')]
            assert trace == repeated
            public_episodes.append({'fixture': fixture, 'trace': trace, 'trace_sha256': digest(trace)})
    atomic_write(OUT / 'public-episodes.jsonl', b''.join(json_bytes(v) + b'\n' for v in public_episodes))
    atomic_write(OUT / 'experiment-config.json', json_bytes(config.model_dump(mode='json')))
    atomic_write(OUT / 'public-catalog.json', json_bytes([v.model_dump(mode='json') for v in env.catalog()]))
    atomic_write(OUT / 'public-fixtures.json', json_bytes(fixtures))
    report = {'status': 'OK', 'generated_at': datetime.now(timezone.utc).isoformat(), 'http_requests': 0,
        'bundle': str(BUNDLE), 'unique_matches': len({f['match_id'] for f in fixtures}), 'episodes': len(fixtures),
        'actions_executed': sum(results.values()), 'results': dict(results), 'final_masks': dict(mask_counts),
        'groups': ['strength', 'recent_form', 'rolling_xg'], 'saved_public_traces': len(public_episodes),
        'deterministic_trace_replay': True, 'public_state_restore': True,
        'research_consumer': 'football_research.acquisition.platform_bridge.to_research_state/to_research_actions',
        'agent_consumer': 'football_agent.research_bridge.platform_state',
        'consumer_actions': to_research_actions(env.catalog()),
        'source_observed_time_is_not_historical_availability': True,
        'experiment_role': 'exploratory', 'costs': 'simulated_reveal_not_actual_HTTP_saving',
        'owner_only_manifest': True, 'public_configuration': env.configuration()}
    atomic_write(OUT / 'report.json', json_bytes(report))
    print(json.dumps({k: report[k] for k in ('status', 'unique_matches', 'episodes', 'actions_executed', 'results', 'final_masks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
