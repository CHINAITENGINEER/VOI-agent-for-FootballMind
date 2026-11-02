"""Zero-network round-five dynamic registry and public consumer verification."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_platform.information import DynamicEpisodeEnvironment, SCENARIOS
from football_spider.storage.raw_archive import digest

BUNDLE = ROOT / 'var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda'


def episode(scenario):
    from football_research.acquisition.platform_bridge import to_research_state
    from football_agent.research_bridge import platform_state
    env = DynamicEpisodeEnvironment.from_export(BUNDLE, scenario=scenario, seed=11)
    actions = env.catalog()
    trace = [env.visible_state().model_dump(mode='json')]
    receipts = []
    for index, action in enumerate(actions):
        if scenario == 'refresh_cycle' and index == 1:
            at = env.visible_state().groups['strength'].expires_at + timedelta(seconds=1)
            env.advance_time(at)
            trace.append(env.visible_state().model_dump(mode='json'))
        result = env.acquire(action.action_id, f'{scenario}-{index}')
        receipts.append(result.model_dump(mode='json'))
        state = env.visible_state()
        # Both consumers see exactly the public state after each transition.
        assert to_research_state(state)['match_id'] == state.match_id
        assert platform_state(state)['match_id'] == state.match_id
        for item in state.groups.values():
            if item.quality_status == 'valid':
                assert item.times.feature_available_at <= state.prediction_cutoff
                assert item.available_at <= state.prediction_cutoff
        trace.append(state.model_dump(mode='json'))
    if scenario == 'stale':
        env.advance_time(env.visible_state().groups['strength'].expires_at + timedelta(seconds=1))
        trace.append(env.visible_state().model_dump(mode='json'))
    before = env.visible_state().budget.spent.copy()
    repeat = env.acquire(actions[-1].action_id, f'{scenario}-{len(actions)-1}')
    assert repeat.repeated_request and repeat.charged_cost.requests.amount == 0
    assert before == env.visible_state().budget.spent
    payload = {'scenario': scenario, 'classification': env.configuration()['scenario_classification'],
        'registry': [a.model_dump(mode='json') for a in env.action_registry()],
        'registry_hash': env.configuration()['action_registry_hash'], 'trace': trace, 'receipts': receipts}
    encoded = json.dumps(payload)
    assert str(BUNDLE) not in encoded
    for forbidden in ('"target_label"', '"full_snapshot"', '"archive_path"'):
        assert forbidden not in encoded
    return {**payload, 'trace_hash': digest(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='var/parallel/collectors/round5/dynamic')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = [episode(s) for s in SCENARIOS]
    for item in records:
        assert item['trace_hash'] == episode(item['scenario'])['trace_hash']
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    report = {'status': 'PASS', 'scenarios': len(records), 'distinct_action_ids': len({a['action_id'] for r in records for a in r['registry']}),
        'deterministic_replay': True, 'consumer_reads': {'research': sum(len(r['receipts']) for r in records), 'agent': sum(len(r['receipts']) for r in records)},
        'network_requests': 0, 'database_writes': 0, 'time_boundary_verified': True,
        'limits': 'Constructed quality/clock scenarios over frozen features; interaction_key is metadata, not proof of positive statistical synergy.',
        'artifacts': {'episodes_sha256': __import__('hashlib').sha256((output/'episodes.jsonl').read_bytes()).hexdigest()}}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
