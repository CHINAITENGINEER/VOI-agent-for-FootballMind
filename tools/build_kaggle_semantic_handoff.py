"""Package public archived bodies for fixed, label-free GPU embedding inference."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.postread_semantic_embedding import EncoderContract, SemanticCache

ROOT = Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def build(output):
    if output.exists():
        raise FileExistsError('Use a new output directory; frozen handoffs are immutable')
    payload = output / 'input'
    (payload / 'texts').mkdir(parents=True)
    (payload / 'controls').mkdir()
    receipt = json.loads(Path('E:/FootballMind/models/qwen3-embedding-0.6b-pinned.json').read_text(encoding='utf-8'))
    contract = EncoderContract()
    long_contract = replace(contract, max_input_tokens=32768, implementation='postread-qwen-full-history-long-v1')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(receipt['model_path'], local_files_only=True, trust_remote_code=False)
    cache_root = Path('E:/FootballMind/embeddings/postread-qwen-frozen-v1')
    caches = {c.fingerprint(): SemanticCache(cache_root, c) for c in (contract, long_contract)}
    source = ROOT / 'data/eligible-encoding-inputs.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').split('\n') if line.strip()]
    grouped = {r['text_sha256']: {'path': Path(r['text_path']), 'group': 'main_historical_corpus'} for r in rows}
    trajectory_root = ROOT / 'recorded-trajectories-v2'
    missing = json.loads((trajectory_root / 'summary.json').read_text(encoding='utf-8'))['primary_library_missing_adopted_text_sha256']
    for key in missing:
        grouped.setdefault(key, {'path': trajectory_root / 'bodies' / (key + '.txt'), 'group': 'recorded_replay_diagnostic'})
    index = []
    references = []
    mapping = []
    for key, spec in sorted(grouped.items()):
        raw = spec['path'].read_bytes()
        text = raw.decode('utf-8')
        if sha(raw) != key or not text.strip():
            raise ValueError('Body SHA or nonempty validation failed: ' + key)
        token_count = len(tok.encode(text, truncation=False))
        active = contract if token_count <= contract.max_input_tokens else long_contract
        if token_count > active.max_input_tokens:
            raise ValueError('Body exceeds declared context; no implicit truncation: ' + key)
        relative = 'texts/' + key + '.txt'
        (payload / relative).write_bytes(raw)
        row = {'text_sha256': key, 'text_path': relative, 'token_count': token_count,
               'contract_fingerprint': active.fingerprint(), 'task_group': spec['group']}
        index.append(row)
        mapping.append({'text_sha256': key, 'original_path': str(spec['path']), 'task_group': spec['group']})
        entry = caches[active.fingerprint()].get(text)
        if entry is not None:
            references.append((row, entry[1], caches[active.fingerprint()].root / (key + '.npy')))
    references.sort(key=lambda x: (x[0]['token_count'], x[0]['text_sha256']))
    if len(references) < 8:
        raise ValueError('Eight independent existing CPU vectors required')
    chosen = [references[round(i * (len(references) - 1) / 7)] for i in range(8)]
    controls = []
    for row, meta, vector_path in chosen:
        key = row['text_sha256']
        rel = 'controls/' + key + '.npy'
        raw = vector_path.read_bytes()
        (payload / rel).write_bytes(raw)
        controls.append({'text_sha256': key, 'vector_path': rel, 'vector_file_sha256': sha(raw),
                         'contract_fingerprint': row['contract_fingerprint'], 'token_count': row['token_count'],
                         'execution_device': 'cpu', 'dtype': 'float32'})
    text_bytes = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in index).encode('utf-8')
    (payload / 'texts.jsonl').write_bytes(text_bytes)
    job = {
        'schema': 'footballmind-kaggle-semantic-job-v1',
        'job_id': 'full-history-qwen-' + sha(text_bytes)[:16],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model': {'repo': contract.repository, 'revision': contract.revision,
                  'files_sha256': {r['file']: r['sha256'] for r in receipt['files'] if r['file'] != 'README.md'}},
        'contracts': {c.fingerprint(): asdict(c) for c in (contract, long_contract)},
        'texts_file': 'texts.jsonl', 'texts_sha256': sha(text_bytes), 'expected_texts': len(index),
        'controls': controls,
        'control_thresholds': {'minimum_cosine': 0.99999, 'maximum_absolute_difference': 0.001},
        'inference_only': True, 'train_encoder': False, 'contains_labels_or_model_targets': False,
        'input_order_is_not_agent_order': True,
        'local_source_index_sha256': sha(source.read_bytes()),
        'notes': ['Full verbatim archived bodies; no title, query, prompt or outcomes are appended.',
                  'CPU and GPU float32 kernels are compared, never assumed bitwise identical.',
                  'Previously generated local CPU cache entries must not be overwritten.',
                  'Recorded replay diagnostic texts remain separate from main training samples.'],
    }
    dump(payload / 'job.json', job)
    dump(output / 'local-input-mapping.json', mapping)
    dump(output / 'package-stats.json', {
        'job_sha256': sha((payload / 'job.json').read_bytes()), 'texts': len(index),
        'groups': dict(Counter(r['task_group'] for r in index)),
        'input_text_bytes': sum((payload / r['text_path']).stat().st_size for r in index),
        'total_tokens': sum(r['token_count'] for r in index),
        'maximum_tokens': max(r['token_count'] for r in index),
        'contract_counts': dict(Counter(r['contract_fingerprint'] for r in index)),
        'controls': len(controls), 'local_existing_vectors_at_snapshot': len(references),
        'model_weights_in_upload': False, 'model_download_bytes': sum(r['bytes'] for r in receipt['files'] if r['file'] != 'README.md'),
        'labels_in_upload': False, 'credentials_in_upload': False,
    })


def notebook():
    cells = []
    def cell(kind, source):
        d = {'cell_type': kind, 'id': sha(source.encode('utf-8'))[:12], 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            d.update(execution_count=None, outputs=[])
        cells.append(d)
    cell('markdown', '''# FootballMind：冻结 Qwen 正文编码\n\n先启用 GPU 和 Internet，将配套 ZIP 作为 **Private Dataset** 添加到 Notebook。运行全部单元。此 Notebook 仅编码正文，不训练 Qwen、不读取赛果、不改变 f/g。正常结束后下载最后一格的 ZIP。\n\n模型为 Qwen/Qwen3-Embedding-0.6B，固定 revision；GPU 与本机都用 float32。保留 GPU 自带 PyTorch，记录实际版本并先做 8 篇 CPU 数值对照。不得为省显存自行截断正文或切换半精度。''')
    cell('code', '%pip install -q "transformers==5.8.0" "huggingface_hub==1.14.0" "safetensors>=0.4.3"\n')
    cell('code', '''from pathlib import Path
import json, subprocess, sys, hashlib, zipfile
import torch
assert torch.cuda.is_available(), "请先在 Notebook Settings 中启用 GPU"
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
jobs = []
for p in Path('/kaggle/input').rglob('job.json'):
    try:
        if json.loads(p.read_text())['schema'] == 'footballmind-kaggle-semantic-job-v1':
            jobs.append(p)
    except (ValueError, KeyError):
        pass
assert len(jobs) == 1, f"应添加一份已解压的配套 Dataset，实际找到 {len(jobs)} 份 job.json"
INPUT = jobs[0].parent
RUNNER = INPUT / 'run_kaggle_semantic_encoding.py'
assert RUNNER.is_file(), '输入 Dataset 中缺少配套 runner'
manifest = json.loads((INPUT / 'payload-manifest.json').read_text())
for rel, expected in manifest['files_sha256'].items():
    assert hashlib.sha256((INPUT / rel).read_bytes()).hexdigest() == expected, rel
OUTPUT = Path('/kaggle/working/footballmind-qwen-output')
print("输入正文数:", json.loads(jobs[0].read_text())['expected_texts'])
print("输入 job SHA:", hashlib.sha256(jobs[0].read_bytes()).hexdigest())
''')
    cell('code', '''subprocess.run([sys.executable, str(RUNNER), '--input-dir', str(INPUT),
                '--output-dir', str(OUTPUT)], check=True)
''')
    cell('code', '''from IPython.display import display, FileLink
completion = json.loads((OUTPUT / 'completion.json').read_text())
display(completion)
ARCHIVE = Path('/kaggle/working/footballmind-qwen-output.zip')
with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUTPUT).as_posix())
print('输出 ZIP SHA-256:', hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())
print('下载后放入 E:/FootballMind/incoming/；由配套回收器校验后使用，不覆盖旧缓存。')
import os
os.chdir('/kaggle/working')
display(FileLink(ARCHIVE.name))
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def finalize(output):
    payload = output / 'input'
    runner = Path(__file__).with_name('run_kaggle_semantic_encoding.py')
    target = payload / runner.name
    if target.exists():
        raise FileExistsError('Do not overwrite finalized handoff')
    shutil.copyfile(runner, target)
    files = {p.relative_to(payload).as_posix(): sha(p.read_bytes()) for p in sorted(payload.rglob('*')) if p.is_file()}
    dump(payload / 'payload-manifest.json', {'schema': 'footballmind-kaggle-payload-v1', 'files_sha256': files})
    dump(output / 'FootballMind_Qwen_GPU.ipynb', notebook())
    archive = output / 'footballmind-qwen-input.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for name in z.namelist():
            if z.read(name) != (payload / name).read_bytes():
                raise ValueError('ZIP bytes differ')
    dump(output / 'delivery.json', {'archive': str(archive), 'bytes': archive.stat().st_size,
         'sha256': sha(archive.read_bytes()), 'zip_bytes_verified': True, 'zip_entries': len(files) + 1,
         'notebook': str(output / 'FootballMind_Qwen_GPU.ipynb'),
         'notebook_sha256': sha((output / 'FootballMind_Qwen_GPU.ipynb').read_bytes()),
         'gpu_execution_performed_locally': False})
    stats = json.loads((output / 'package-stats.json').read_text(encoding='utf-8'))
    instructions = f'''FootballMind Kaggle GPU 编码交接

本包已在本机准备；尚未在你的 Kaggle 账户执行。只做冻结模型前向推理，不训练 Qwen，也不改变 f/g 或既有比赛划分。

上传文件
1. footballmind-qwen-input.zip：{len(files)+1} 个文件，{archive.stat().st_size / 1024**2:.2f} MiB。
2. FootballMind_Qwen_GPU.ipynb：可导入 Kaggle 的 Notebook。

操作
1. 在自己的 Kaggle 账户创建 Private Dataset，上传上述 ZIP。确认解压后能看到 job.json、texts.jsonl、texts/、controls/ 和 runner。
2. 新建 Notebook，导入 FootballMind_Qwen_GPU.ipynb，将这个 Private Dataset 加入 Notebook 的 Input。
3. 在 Settings 中选择可用的 GPU，并启用 Internet。公开模型权重将按固定 revision 下载，约 1.21 GB；不需要 DeepSeek 或其他 LLM API Key。
4. Run All。首批会核对 8 篇 CPU 参考向量，全部通过后继续；每篇向量立即保存，可在同输出目录恢复。显存不足等错误会明确停止，不会偷偷截断或改变精度。
5. 完成后从 Output 下载 footballmind-qwen-output.zip。建议 Save Version 保留此次输出；不要只关闭页面而不保存或下载。
6. 将下载文件放入 E:/FootballMind/incoming/footballmind-qwen-output.zip，告知主线程即可。不要覆盖原本地缓存或旧训练结果。

本包范围
正文 {stats['texts']} 篇：主库 3001 篇，另有真实轨迹回放诊断 5 篇。
总 tokenizer tokens：{stats['total_tokens']}，最长：{stats['maximum_tokens']}。
原生输出 1024 维，float32，固定 last-nonpadding pooling + L2。训练端按既定协议取256维前缀再L2；本包不改训练端维度。
全文逐字节保留，无标题注入、prompt 或截断；没有赛果/损失/gain 标签文件，没有凭据。
模型 Qwen/Qwen3-Embedding-0.6B，revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3。
CPU/GPU可能存在浮点差异；8个参照以及回收时所有可用CPU重叠项均检查，不声称逐位一致。

本机回收命令（可交给主线程执行）
python tools/import_kaggle_semantic_encoding.py --input-dir "{payload.as_posix()}" --result-zip "E:/FootballMind/incoming/footballmind-qwen-output.zip" --output-dir "{(ROOT / 'gpu-import-v1').as_posix()}" --cpu-cache "E:/FootballMind/embeddings/postread-qwen-frozen-v1"

回收器先校验完整性及数值对照，再向新目录写出 embeddings.npz；既有CPU向量优先保留，不向运行中的CPU缓存写入。
如果 Kaggle 无法下载固定模型，不要换模型。可以另传本地同 revision 的模型文件，并使用 runner 的 --model-dir 参数；具体由主线程协助处理。

官方操作说明：https://www.kaggle.com/docs/notebooks
'''
    (output / 'README.txt').write_text(instructions, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--finalize', action='store_true')
    args = p.parse_args()
    (finalize if args.finalize else build)(args.output)
