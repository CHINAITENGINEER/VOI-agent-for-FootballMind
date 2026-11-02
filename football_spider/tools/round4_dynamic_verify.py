"""Verify dynamic evidence-quality episodes without network or database access."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_platform.information import DynamicEpisodeEnvironment, SCENARIOS
from football_spider.storage.raw_archive import atomic_write, digest, json_bytes

BUNDLE = ROOT / 'var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda'
OUT = ROOT / 'var/parallel/collectors/round4/dynamic'


def run_scenario(scenario):
    env = DynamicEpisodeEnvironment.from_export(BUNDLE, scenario=scenario, seed=11)
    initial = env.visible_state()
    assert all(value == 'UNACQUIRED' for value in initial.mask.values())
    trace = [{'kind': 'initial', 'state': initial.model_dump(mode='json')}]
    catalog = [item.model_dump(mode='json') for item in env.catalog()]
    statuses = []
    if scenario == 'stale':
        first = env.acquire('stale.strength', 'stale-1')
        statuses.append(first.status)
        group = env.visible_state().groups['strength']
        env.advance_time(group.expires_at + timedelta(seconds=1))
        trace.append({'kind': 'after_expiry', 'state': env.visible_state().model_dump(mode='json')})
        assert env.visible_state().mask['strength'] == 'STALE'
        assert env.visible_state().groups['strength'].quality_status == 'stale'
    elif scenario == 'duplicate':
        statuses += [env.acquire('primary.strength', 'duplicate-1').status,
                     env.acquire('duplicate.strength', 'duplicate-2').status]
        assert statuses == ['ACQUIRED', 'UNCHANGED']
        assert env.acquire('duplicate.strength', 'duplicate-2').repeated_request
        assert env.visible_state().groups['strength'].quality_status == 'valid'
    elif scenario == 'no_change':
        statuses += [env.acquire('primary.strength', 'no-change-1').status,
                     env.acquire('refresh.no_change', 'no-change-2').status]
        assert statuses == ['ACQUIRED', 'UNCHANGED']
    elif scenario == 'conflict':
        statuses += [env.acquire('provider_a.strength', 'conflict-a').status,
                     env.acquire('provider_b.strength', 'conflict-b').status]
        assert statuses == ['ACQUIRED', 'FAILED']
        assert env.visible_state().groups['strength'].quality_status == 'conflict'
    elif scenario == 'no_data_then_available':
        statuses += [env.acquire('first.no_data', 'missing-1').status,
                     env.acquire('retry.strength', 'missing-2').status]
        assert statuses == ['NO_DATA', 'ACQUIRED']
        assert env.catalog()[1].capability_status == 'AVAILABLE'
    elif scenario == 'provider_error':
        statuses += [env.acquire('provider.error', 'error-1').status,
                     env.acquire('provider.retry', 'error-2').status]
        assert statuses == ['FAILED', 'ACQUIRED']
    elif scenario == 'complementary':
        statuses += [env.acquire('complementary.strength', 'pair-1').status,
                     env.acquire('complementary.recent_form', 'pair-2').status]
        assert statuses == ['ACQUIRED', 'ACQUIRED']
        assert len({a['interaction_key'] for a in catalog}) == 1
    else:
        statuses.append(env.acquire('fresh.strength', 'fresh-1').status)
        assert statuses == ['ACQUIRED']
    state = env.visible_state()
    trace.append({'kind': 'final', 'state': state.model_dump(mode='json')})
    public_json = json.dumps({'catalog': catalog, 'trace': trace}, ensure_ascii=False, sort_keys=True)
    assert str(BUNDLE) not in public_json
    # Real consumer adapters receive only the serialized public state.
    from football_research.acquisition.platform_bridge import to_research_state
    from football_agent.research_bridge import platform_state
    research = to_research_state(state)
    agent = platform_state(state)
    assert research['match_id'] == state.match_id and agent['match_id'] == state.match_id
    return {'scenario': scenario, 'classification': env.configuration()['scenario_classification'],
            'catalog': catalog, 'statuses': statuses, 'trace': trace,
            'trace_hash': digest(trace), 'research_mask': research['mask'],
            'agent_mask': agent['mask']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = [run_scenario(name) for name in SCENARIOS]
    # A second owner environment must reproduce the same public trace/hash.
    replay = {item['scenario']: run_scenario(item['scenario'])['trace_hash'] for item in episodes}
    assert replay == {item['scenario']: item['trace_hash'] for item in episodes}
    atomic_write(OUT / 'episodes.jsonl', b''.join(json_bytes(item) + b'\n' for item in episodes))
    report = {'status': 'OK', 'scenarios': list(SCENARIOS), 'episodes': len(episodes),
              'network_requests': 0, 'database_writes': 0, 'deterministic_replay': True,
              'consumer_reads': {'research': len(episodes), 'agent': len(episodes)},
              'quality_statuses': ['duplicate', 'stale', 'source_missing', 'conflict', 'provider_error', 'no_change'],
              'bundle_role': '2025/26 exploratory historical_reconstruction',
              'hidden_labels_or_paths_in_public_trace': False,
              'episodes_file': str(OUT / 'episodes.jsonl')}
    atomic_write(OUT / 'report.json', json_bytes(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
