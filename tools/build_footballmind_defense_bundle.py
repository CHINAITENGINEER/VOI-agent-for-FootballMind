"""Build an independent offline frozen research bundle; never train or fetch."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').split('\n') if line.strip()]


def save(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def copy(source, target):
    if target.exists(): raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(source) != sha(target): raise ValueError('Copy differs')


def main():
    import numpy as np
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--output', type=Path, default=ROOT / 'var/releases/footballmind-defense-20261025-rc1')
    args = parser.parse_args(); src = args.experiment; out = args.output
    if (out / 'manifest.json').exists(): raise FileExistsError('Release already frozen')
    if not (out / 'footballmind_defense.py').is_file(): raise FileNotFoundError('Standalone app scaffold required')
    for folder in ['models/f', 'models/g', 'data/bodies', 'settlement', 'provenance', 'src/fm_defense_core']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    fdir = src / 'f-models/selected-set_residual_mlp'; gdir = src / 'g-models/set'
    fsha = sha(fdir / 'model.pt'); gsha = sha(gdir / 'g-models.joblib')
    if fsha != 'b36b2cf87f379034224c314794efc6ec59266dd4cf4752383965c41fb63661e0':
        raise ValueError('Unexpected f selected checkpoint')
    g = joblib.load(gdir / 'g-models.joblib')
    if g['f_model_sha256'] != fsha: raise ValueError('f/g binding differs')
    frozen = read(gdir / 'selection-frozen.json')
    if frozen['selected']['model'] != 'state_ridge': raise ValueError('Development-selected policy changed')
    configs = read(gdir / 'model.json')['configurations']
    models = {}
    for name, spec in g['fitted'].items():
        choices = [c for c in frozen['all_dev_policy_choices'] if c['model'] == name]
        best = min(choices, key=lambda r: (r['mean_final_log_loss'], r['mean_updates'], r['model'], r['threshold']))
        models[name] = {'model_name': name, 'threshold': best['threshold'], 'budget': best['budget'],
                        'role': 'development_selected_policy' if name == 'state_ridge' else 'research_comparison_head',
                        'input': spec.get('input', 'constant'), 'candidate_body_embedding_used': spec.get('input') == 'full',
                        'configuration': configs[name], 'threshold_basis': 'already-saved g_dev policy results; no new tuning'}
    presentation = dict(models['fulltext_mlp_seed20261027'], role='presentation_fulltext_research_candidate',
                        choice_basis='best previously recorded g_dev gain MSE among three MLP seeds; not overall best policy')
    models['fixed_order'] = {'model_name': 'fixed_order', 'role': 'explicit_non_g_baseline', 'input': 'source_order',
                             'budget': 3, 'threshold': None, 'candidate_body_embedding_used': False}
    models['all_evidence'] = {'model_name': 'all_evidence', 'role': 'explicit_non_g_baseline', 'input': 'source_order_all',
                              'budget': 3, 'threshold': None, 'candidate_body_embedding_used': False}
    if presentation['threshold'] != .01: raise ValueError('Unexpected frozen MLP27 threshold')
    for name in ['model.pt', 'model.json']:
        copy(fdir / name, out / 'models/f' / name)
    for name in ['g-models.joblib', 'model.json', 'selection-frozen.json', 'feature-schema.json']:
        copy(gdir / name, out / 'models/g' / name)
    modules = ['sequential_evidence_f.py', 'sequential_postread_g.py', 'full_history_sequence_metadata.py', 'postread_semantic_embedding.py']
    for name in modules:
        copy(ROOT / 'football_research/acquisition' / name, out / 'src/fm_defense_core' / name)
    (out / 'src/fm_defense_core/__init__.py').write_text('"""Vendored frozen research inference; no production runtime."""\n', encoding='utf8')
    match_by = {r['physical_match_id']: r for r in rows(src / 'data/matches.jsonl')}
    docs = {r['document_id']: r for r in rows(src / 'data/documents.jsonl')}
    sequences = [r for r in rows(src / 'data/sequences.jsonl') if r['eligible'] and r['split_role'] == 'g_dev']
    sequences.sort(key=lambda r: (match_by[r['physical_match_id']]['match_date'], r['physical_match_id']))
    if len(sequences) != 160: raise ValueError('Expected complete preselected 160-match development panel')
    text_paths = {r['text_sha256']: Path(r['text_path']) for r in rows(src / 'data/eligible-encoding-inputs.jsonl')}
    outcome_by = {r['physical_match_id']: r for r in rows(src / 'data/outcomes.jsonl')}
    cases = []; text_shas = set(); document_fields = ['document_id', 'text_sha256', 'source_url', 'source_domain', 'published_at',
                     'captured_at', 'source_available_at', 'historical_version_unknown', 'body_characters', 'existing_claims']
    labels = {}
    for seq in sequences:
        mid = seq['physical_match_id']; match = match_by[mid]
        contexts = {d['document_id']: d.get('metadata_context') or {} for d in seq['documents']}
        case_documents = []
        for did in seq['document_ids']:
            original = docs[did]; merged = dict(original, **contexts[did]); h = original['text_sha256']
            p = text_paths[h]; raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != h: raise ValueError('Exact body byte binding differs: ' + h)
            relative = 'data/bodies/' + h + '.txt'
            if h not in text_shas: copy(p, out / relative)
            text_shas.add(h)
            d = {k: merged.get(k) for k in document_fields};d['body_file'] = relative
            d['source_type'] = merged.get('source_type');d['target_binding'] = merged.get('target_binding')
            d['title'] = next((v.get('title') for v in original.get('source_versions', []) if v.get('title')), None)
            d['body_characters'] = len(raw.decode('utf8'))
            case_documents.append(d)
        labels[mid] = outcome_by[mid]
        cases.append({'case_id': mid, 'physical_match_id': mid, 'split_role': 'g_dev',
                      'match': {k: match.get(k) for k in ['home_team', 'away_team', 'division', 'season', 'match_date', 'kickoff']},
                      'prior': match['prior'], 'prior_provenance': {k: match.get(k) for k in ['prior_source', 'prior_time_verified']},
                      'documents': case_documents, 'source_order_basis': seq['order_basis'],
                      'settlement_available': labels[mid].get('outcome_class') is not None})
    save(out / 'data/cases.json', cases);save(out / 'settlement/outcomes.json', labels)
    with np.load(src / 'gpu-import-v1/embeddings.npz', allow_pickle=False) as archive:
        np.savez_compressed(out / 'data/embeddings.npz', **{h: archive[h] for h in sorted(text_shas)})
    encoder = read(src / 'f-fit-vectors/encoder-provenance.json')
    save(out / 'provenance/encoder-provenance.json', encoder)
    copy(src / 'f-models/selection-freeze.json', out / 'provenance/f-selection-freeze.json')
    copy(gdir / 'metrics.json', out / 'provenance/g-recorded-metrics.json')
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']}
    config = {'schema': 'footballmind-defense-frozen-bundle-v1', 'bundle_id': out.name,
              'f_model_path': 'models/f', 'g_model_path': 'models/g/g-models.joblib', 'g_model_sha256': gsha,
              'f_model_sha256': fsha, 'presentation_default': presentation,
              'development_selected_policy': models['state_ridge'], 'models': models,
              'dependency_versions': versions, 'python_version': platform.python_version(),
              'encoder': {'model': 'Qwen/Qwen3-Embedding-0.6B', 'revision': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
                          'native_dimension': 1024, 'prefix_dimension': 256, 'text_policy': 'full body; frozen cached vectors, no runtime encoding'},
              'case_selection': 'all 160 g_dev matches, sorted match_date then physical_match_id; no outcome filtering',
              'cases': len(cases), 'unique_bodies': len(text_shas), 'default_case_id': cases[0]['case_id'],
              'research_only': True, 'production_enabled': False,
              'limitations': ['历史归档研究演示，非实时未来比赛预测。',
                              '默认全文MLP是研究候选，未证实优于随机或基线；阈值0.01及预算3沿用原开发集冻结值。',
                              'state_ridge是原开发集完整策略优选，仅按当前状态估计采用/停止；同状态候选同分，按冻结来源顺序破平，不是全文排序模型。',
                              '已保存正文全部可供g读取，f仅使用所选正文；正文与向量获取成本已发生，不能声称由g节省。',
                              '历史正文版本及报价时钟存在未知，来源时间不补造；赛果只在显式结算层使用。',
                              '模型原始metadata中的绝对路径仅保留血缘，不参与便携推理；所有运行所需文件在本包内。']}
    save(out / 'bundle-config.json', config)
    (out / 'requirements-lock.txt').write_text('\n'.join(k+'=='+v for k,v in versions.items())+'\n',encoding='utf8')
    source_bindings = {str(p): sha(p) for p in [fdir/'model.pt',gdir/'g-models.joblib',src/'data/sequences.jsonl',src/'gpu-import-v1/embeddings.npz']}
    save(out / 'provenance/build-source-bindings.json', {'source_files': source_bindings, 'build_script_sha256': sha(__file__),
            'source_paths_required_for_inference': False, 'new_training': False, 'network_requests': 0})
    (out / 'README.txt').write_text('FootballMind 答辩冻结研究包\n\n'
        '运行目录可移动，依赖版本见 requirements-lock.txt。不会自动下载模型或访问网络。\n'
        'python footballmind_defense.py --command verify\n'
        'python footballmind_defense.py --command catalog\n'
        'python footballmind_defense.py --command run --output run.json\n'
        'python footballmind_defense.py --command run --model state_ridge --settle --output comparison.json\n\n'
        '默认set f+fulltext_mlp_seed20261027，阈值0.01，预算3，为全文研究候选。\n'
        '原开发集完整策略优选state_ridge阈值0也保留；它不按候选全文区分同状态文章，不声称最佳全文筛选。\n'
        '仅加载本地已有正文和对应Qwen向量，f/g重新计算真实分数及概率；不是播放旧预测JSON。\n'
        '默认选择开发组按日期及比赛ID排序的第一场，没有按胜负或gain选案例。\n'
        'run_case/catalog/get_case不加载赛果字段；显式settle_run才读取settlement/outcomes.json。\n'
        'manifest验证失败、缺失正文/向量、未知模型、依赖不符均报错，无随机假数据或固定概率fallback。\n', encoding='utf8')
    manifest = {str(p.relative_to(out)).replace('\\','/'): sha(p) for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    save(out / 'manifest.json', {'bundle_id': out.name, 'schema': 'footballmind-defense-bundle-manifest-v1', 'files_sha256': manifest})
    print(json.dumps({'output': str(out), 'cases': len(cases), 'bodies':len(text_shas), 'files':len(manifest),
                      'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()), 'f_sha':fsha,'g_sha':gsha,
                      'presentation':presentation, 'manifest_sha256':sha(out/'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__': main()
