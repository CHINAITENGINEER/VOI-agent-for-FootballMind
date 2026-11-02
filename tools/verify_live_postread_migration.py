"""Zero-network replay of real recovered bodies through the frozen live adapter.

This is an offline migration check, not a fresh prospective run or training data.
Inputs are the saved identity/market initialization and exact repaired bodies.
No outcome, old gain, old prediction-after, or semantic API is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN = ROOT / 'var/parallel/agent/round2/runs/live-g/runs/lg-ce4e02daacf14f6aa16af076e9cea394'
RECOVERED = ROOT / 'var/repairs/live-venezia-body-20261025-v1'
OUTPUT = ROOT / 'var/work/live-postread-migration-20261025-v1/offline-validation'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    if 'settlement' in Path(path).parts or Path(path).name in {'outcomes.json', 'outcomes.jsonl', 'run.json'}:
        raise ValueError('This verification does not load outcome files or the old run result')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def aware(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Source clock is not timezone-aware')
    return result


@contextmanager
def no_network():
    blocked = []

    def deny(*args, **kwargs):
        blocked.append({'at': now(), 'operation': 'socket_network_attempt'})
        raise RuntimeError('Network is disabled in the offline migration verification')

    with patch('socket.socket.connect', deny), patch('socket.socket.connect_ex', deny), patch('socket.create_connection', deny), patch('socket.getaddrinfo', deny):
        yield blocked


def frozen_state(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.f.network.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf-8'))
        h.update(str(array.dtype).encode('ascii'))
        h.update(str(array.shape).encode('ascii'))
        h.update(array.tobytes())
    import joblib
    return {
        'f_file_sha256': digest_file(model.release_dir / model.backend.config['f_model_path'] / 'model.pt'),
        'g_file_sha256': digest_file(model.release_dir / model.backend.config['g_model_path']),
        'f_in_memory_state_sha256': h.hexdigest(),
        'selected_g_in_memory_joblib_hash': joblib.hash(model.g['fitted'][model.head]),
        'f_training': bool(model.f.network.training),
        'f_trainable_parameters': sum(p.numel() for p in model.f.network.parameters() if p.requires_grad),
    }


def prepare_inputs(run, recovered):
    identity_path = run / 'identity/identity.json'
    initialization_path = run / 'setup/espn-initialization.json'
    identity = read_json(identity_path)
    market = read_json(initialization_path)['market_initialization']
    saved = identity['canonical']
    fixture = {
        'physical_match_id': saved['source_event_id'],
        'physical_identity_level': identity['identity_level'],
        'independently_confirmed': identity['independently_confirmed'],
        'home_team': saved['home_name'], 'away_team': saved['away_name'],
        'home_aliases': identity['aliases']['home'], 'away_aliases': identity['aliases']['away'],
        'home_aliases_complete': False, 'away_aliases_complete': False,
        'kickoff': saved['kickoff'], 'prediction_cutoff': market['prediction_cutoff'],
        'competition_name': saved['competition_name'], 'season': saved['season'],
    }
    assert market['fixture']['match_id'] == fixture['physical_match_id']
    assert market['fixture']['home_team_name'] == fixture['home_team']
    assert market['fixture']['away_team_name'] == fixture['away_team']
    assert market['fixture']['kickoff'] == fixture['kickoff']
    prior = market['probabilities']
    assert len(prior) == 3 and all(0 < float(v) < 1 for v in prior) and abs(sum(prior) - 1) < 1e-12
    assert aware(market['source_observed_at']) < aware(fixture['prediction_cutoff']) < aware(fixture['kickoff'])
    report_path = recovered / 'replay-report.json'
    repair = read_json(report_path)
    from demo.live_sources import DIRECTORIES
    # This name is deliberately verified against the explicit source registry,
    # rather than guessed from a hostname or inferred from article prose.
    registered = DIRECTORIES.get(fixture['away_team'])
    documents, evidence, input_paths = [], [], [identity_path, initialization_path, report_path]
    for row in repair['records']:
        path = recovered / row['body_file']
        text = path.read_text(encoding='utf-8')
        assert digest_bytes(text.encode('utf-8')) == row['body_sha256']
        assert len(text) == row['body_characters']
        observation_path = ROOT / row['raw_observation_file']
        observation = read_json(observation_path)
        raw_path = run / 'sources' / observation['payload_path']
        payload = gzip.decompress(raw_path.read_bytes())
        assert digest_bytes(payload) == row['raw_sha256'] == observation['content_hash']
        assert observation['endpoint'] == row['source_url'] and observation['http_status'] == 200
        assert aware(row['source_published_at']) <= aware(row['source_response_observed_at']) < aware(fixture['prediction_cutoff'])
        assert registered and urlsplit(registered).hostname == urlsplit(row['source_url']).hostname
        for segment in row['segments']:
            assert text[segment['start']:segment['end']] == segment['text']
        documents.append({
            'document_id': row['document_id'], 'action_id': row['action_id'],
            'text': text, 'text_sha256': row['body_sha256'], 'title': row['title'],
            'source_url': row['source_url'], 'source_domain': urlsplit(row['source_url']).hostname,
            'source_type': 'official_club', 'source_owner_team': fixture['away_team'],
            'source_owner_basis': 'explicit DIRECTORIES registry, exact host',
            'published_at': row['source_published_at'], 'source_modified_at': row['source_modified_at'],
            'captured_at': None, 'source_response_observed_at': row['source_response_observed_at'],
            'source_available_at': row['source_response_observed_at'],
            'source_available_at_basis': 'this exact archived version was received; not first publication',
            'first_public_available_at': None, 'historical_version_unknown': False,
            'existing_claims': None,
        })
        evidence.append({**row, 'body_path': str(path), 'raw_payload_path': str(raw_path),
                         'raw_and_body_sha_verified': True, 'paragraph_offsets_verified': True,
                         'source_observed_before_cutoff_and_kickoff': True})
        input_paths.extend([path, observation_path, raw_path])
    assert len(documents) == 3
    return fixture, prior, documents, evidence, input_paths, {
        'identity': str(identity_path), 'market_initialization': str(initialization_path),
        'market_prior': market['quote_snapshot'], 'prior_source_observed_at': market['source_observed_at'],
        'original_run_result_json_read': False, 'outcome_file_values_read': False,
        'release_integrity_note': 'Bundle manifest verification may hash settlement file bytes; settlement values are never parsed or used.',
    }


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(out / 'status.json', {'status': 'preparing', 'started_at': started, 'scope': 'offline_replay'})
    fixture, prior, documents, evidence, input_paths, origins = prepare_inputs(args.run.resolve(), args.recovered.resolve())
    inputs_before = {str(path): digest_file(path) for path in input_paths}
    write_json(out / 'inputs.json', {'fixture': fixture, 'market_prior': prior, 'documents': documents, 'origins': origins})
    write_json(out / 'source-evidence.json', evidence)
    from demo.live_postread_model import LivePostreadModel
    with no_network() as blocked:
        model = LivePostreadModel(cache_root=args.cache_root)
        import torch
        torch.set_num_threads(1)
        fixed = frozen_state(model)
        assert not fixed['f_training'] and fixed['f_trainable_parameters'] == 0
        write_json(out / 'model-bindings.json', model.bindings)
        encoded, vectors, encodings = [], [], []
        for i, doc in enumerate(documents):
            stamp = now()
            begin = time.perf_counter()
            result = model.encode_document(doc)
            encoded.append(result['document'])
            vectors.append(result['vector'])
            entry = {'document_id': doc['document_id'], 'encoding_started_at': stamp,
                     'encoding_finished_at': now(), 'elapsed_seconds': time.perf_counter() - begin,
                     'vector_value_sha256': digest_bytes(result['vector'].tobytes()),
                     'vector_file_sha256': digest_file(result['encoding']['vector_path']),
                     'l2_norm': float(np.linalg.norm(result['vector'])), **result['encoding']}
            encodings.append(entry)
            write_json(out / 'encoding-provenance.json', encodings)
            assert frozen_state(model) == fixed
            print(json.dumps({'encoded': i + 1, 'document_id': doc['document_id'], 'cache_hit': entry['cache_hit'], 'seconds': entry['elapsed_seconds']}), flush=True)
        initial = model.initial_prediction(fixture, prior)
        pool_initial = model.predict(fixture, prior, encoded, vectors, [])
        assert initial == pool_initial, 'Unadopted candidate pool must not change f state'
        all_steps, adopted, previous = [], [], initial
        for index in range(len(encoded)):
            before = model.predict(fixture, prior, encoded, vectors, adopted)
            assert before == previous
            adopted.append(index)
            after = model.predict(fixture, prior, encoded, vectors, adopted)
            assert after['used_input_ids'] == [encoded[i]['document_id'] for i in adopted]
            assert frozen_state(model) == fixed
            all_steps.append({'step': len(adopted), 'selection_policy': 'fixed_order_all_evidence_control',
                'selected_document_id': encoded[index]['document_id'], 'selected_action_id': encoded[index]['action_id'],
                'selected_body_sha256': encoded[index]['text_sha256'], 'f_executed': True,
                'prediction_before': before['probabilities'], 'prediction_after': after['probabilities'],
                'probability_l1_change': float(np.abs(np.array(after['probabilities']) - before['probabilities']).sum()),
                'used_input_ids_before': before['used_input_ids'], 'used_input_ids_after': after['used_input_ids'],
                'frozen_parameters': fixed, 'recorded_at': now()})
            previous = after
        write_json(out / 'all-evidence-control.json', {'initial_prediction': initial, 'steps': all_steps,
                    'final_prediction': previous, 'body_consumption_count': len(all_steps),
                    'g_selection_executed': False, 'outcomes_used': False})
        decisions, adopted = [], []
        while True:
            decision = model.evaluate(fixture, prior, encoded, vectors, adopted, budget=3, cost_loss_units=0.0)
            before = model.predict(fixture, prior, encoded, vectors, adopted)
            assert decision['prediction'] == before
            index = decision['selected_index']
            entry = {'decision_number': len(decisions) + 1, **decision,
                     'f_body_update_executed': False, 'prediction_after': None,
                     'used_input_ids_before': before['used_input_ids'], 'used_input_ids_after': None,
                     'recorded_at': now()}
            if index is not None:
                adopted.append(index)
                after = model.predict(fixture, prior, encoded, vectors, adopted)
                assert after['used_input_ids'] == [encoded[i]['document_id'] for i in adopted]
                entry.update(f_body_update_executed=True, prediction_after=after['probabilities'],
                             used_input_ids_after=after['used_input_ids'], selected_body_sha256=encoded[index]['text_sha256'],
                             probability_l1_change=float(np.abs(np.array(after['probabilities']) - before['probabilities']).sum()))
            assert frozen_state(model) == fixed
            entry['frozen_parameters'] = fixed
            decisions.append(entry)
            if index is None:
                break
            assert len(decisions) <= 3
        final = model.predict(fixture, prior, encoded, vectors, adopted)
        autonomous = {'initial_prediction': initial, 'decisions': decisions, 'final_prediction': final,
                      'adopted_document_ids': final['used_input_ids'], 'body_consumption_count': len(adopted),
                      'stop_reason': decisions[-1]['stop_reason'], 'threshold': model.threshold, 'budget': 3,
                      'cost_loss_units': 0.0, 'threshold_adjusted': False, 'outcomes_used': False}
        write_json(out / 'autonomous-g-replay.json', autonomous)
        assert not blocked
    inputs_after = {str(path): digest_file(path) for path in input_paths}
    assert inputs_before == inputs_after
    summary = {'schema': 'footballmind-live-postread-migration-verification-v1', 'status': 'passed',
               'scope': 'offline_replay_of_archived_real_bodies; not a newly executed live Agent trajectory',
               'started_at': started, 'finished_at': now(), 'fixture': fixture,
               'body_count': len(encoded), 'body_characters': [len(d['text']) for d in encoded],
               'real_local_encodings': sum(not e['cache_hit'] for e in encodings),
               'exact_text_cache_hits': sum(e['cache_hit'] for e in encodings),
               'all_evidence_f_consumptions': len(all_steps), 'autonomous_f_consumptions': len(adopted),
               'autonomous_stop_reason': autonomous['stop_reason'], 'autonomous_decisions': len(decisions),
               'raw_market_prior': prior, 'new_f_initial_probabilities': initial['probabilities'],
               'all_evidence_final_probabilities': previous['probabilities'], 'autonomous_final_probabilities': final['probabilities'],
               'bindings': model.bindings, 'frozen_parameters_before_after_equal': True,
               'unadopted_pool_does_not_change_f_initial_state': True, 'exact_used_ids_body_bindings_verified': True,
               'source_inputs_unchanged': True, 'input_sha256': inputs_before,
               'network_requests': 0, 'network_attempts_blocked': len(blocked), 'external_api_requests': 0,
               'training_calls': 0, 'outcome_values_used': False, 'loss_or_gain_labels_generated': False,
               'empirical_training_rows_generated': 0,
               'limits': ['Original identity is source-confirmed ESPN, not independent official confirmation.',
                          'This run tests migration on the already recovered three-body pool; it does not test new HTTP discovery.',
                          'Probability movement is not accuracy improvement; no result or loss is computed.',
                          'Live metadata historical_reconstruction is zero; source is archived and this replay remains offline.',
                          'Claims remain unknown; there is no semantic provider or synthesized structured claim.',
                          'Policy uses existing frozen developer-selected threshold; no forced selection.',
                          'Current model is research-only and cross-league transfer has not been calibrated.']}
    write_json(out / 'verification.json', summary)
    lines = ['真实正文迁移离线验收：通过', f"比赛：{fixture['home_team']} – {fixture['away_team']} ({fixture['physical_match_id']})",
             '本次只离线复用已归档正文；没有 HTTP、外部 API、赛果读取、训练或 empirical 标签。',
             f'三篇正文字符数：{summary["body_characters"]}；实际本地新编码 {summary["real_local_encodings"]}，精确缓存命中 {summary["exact_text_cache_hits"]}。',
             f'原始市场先验：{prior}', f'新 f 初始概率：{initial["probabilities"]}',
             f'全部采用对照：{len(all_steps)} 次真实 f 正文更新；最终概率 {previous["probabilities"]}。',
             f'g 自主策略：采用 {len(adopted)} 篇，停止原因 {autonomous["stop_reason"]}，冻结阈值 {model.threshold}。',
             f'g 自主最终概率：{final["probabilities"]}',
             '各步 used_input_ids 与正文 SHA 一一绑定，冻结 f/g 文件与内存参数始终一致。',
             'STOP 没有新正文更新：该步 prediction_after=null；初始 f 计算与正文更新分别记录。',
             '概率变化仅证明模型响应，不代表预测改善。新实时 API 的网络获取需另外验收。']
    (out / 'readable-report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'status.json', {'status': 'passed', 'started_at': started, 'finished_at': now()})
    write_json(out / 'manifest.json', {'schema': 'footballmind-file-manifest-v1',
        'script': str(Path(__file__).resolve()), 'script_sha256': digest_file(__file__),
        'files': [{'path': str(p.relative_to(out)), 'sha256': digest_file(p), 'bytes': p.stat().st_size}
                  for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json']})
    print(json.dumps({k: summary[k] for k in ('status', 'body_count', 'real_local_encodings', 'all_evidence_f_consumptions', 'autonomous_f_consumptions', 'autonomous_stop_reason', 'network_requests')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--recovered', type=Path, default=RECOVERED)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', {'status': 'failed', 'at': now(),
                       'exception_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
