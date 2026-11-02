"""Ship an embedded, SHA-bound runner repair without re-uploading the dataset."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import zlib


ROOT = Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')
OUTPUT = ROOT / 'kaggle-t4-repair-v2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    job_sha = digest((ROOT / 'kaggle-handoff-v1/input/job.json').read_bytes())
    embedded = {}
    for name in ('run_kaggle_semantic_encoding.py', 'run_kaggle_semantic_encoding_v2.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        embedded[name] = {'sha256': digest(raw), 'zlib_base64': base64.b64encode(zlib.compress(raw, 9)).decode('ascii')}
        (OUTPUT / name).write_bytes(raw)
    bootstrap = '''from pathlib import Path
import base64, hashlib, json, subprocess, sys, zlib, zipfile
import torch
assert torch.cuda.is_available(), '需要 GPU，请保持 Kaggle 的 T4 配置'
'''
    bootstrap += 'EXPECTED_JOB_SHA = ' + repr(job_sha) + '\n'
    bootstrap += 'EMBEDDED_RUNNERS = ' + repr(embedded) + '\n'
    bootstrap += '''PATCH_DIR = Path('/kaggle/working/footballmind-runner-v2')
PATCH_DIR.mkdir(parents=True, exist_ok=True)
for name, spec in EMBEDDED_RUNNERS.items():
    raw = zlib.decompress(base64.b64decode(spec['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == spec['sha256'], name
    destination = PATCH_DIR / name
    if destination.exists():
        assert destination.read_bytes() == raw, '已有不同版本修复文件，请勿覆盖：' + name
    else:
        destination.write_bytes(raw)
old_input = globals().get('INPUT')
if old_input is not None and (Path(old_input) / 'job.json').is_file() and hashlib.sha256((Path(old_input) / 'job.json').read_bytes()).hexdigest() == EXPECTED_JOB_SHA:
    INPUT = Path(old_input)
else:
    jobs = [p for p in Path('/kaggle/input').rglob('job.json') if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_JOB_SHA]
    assert jobs, '请将原正文 Dataset 添加为 Input；无需重新上传正文'
    INPUT = sorted(jobs)[0].parent
OUTPUT = Path('/kaggle/working/footballmind-qwen-output')
RUNNER_V2 = PATCH_DIR / 'run_kaggle_semantic_encoding_v2.py'
print('GPU:', torch.cuda.get_device_name(0), '可见 GPU 数:', torch.cuda.device_count())
print('正文输入:', INPUT)
print('继续使用输出:', OUTPUT)
print('检测到已保存向量文件:', len(list(OUTPUT.glob('*.npy'))), '；runner 会校验完整元数据后才复用。')
print('注意力按256个query分块；保持完整K/V、float32和原文。此次使用cuda:0。')
'''
    execution = '''subprocess.run([sys.executable, str(RUNNER_V2), '--input-dir', str(INPUT),
                '--output-dir', str(OUTPUT), '--device', 'cuda:0',
                '--query-chunk-size', '256'], check=True)
'''
    download = '''completion = json.loads((OUTPUT / 'completion.json').read_text())
assert completion['status'] == 'completed', completion
print('完成:', completion['completed_texts'], '复用已有:', completion['resumed_texts'])
ARCHIVE = Path('/kaggle/working/footballmind-qwen-output.zip')
with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUTPUT).as_posix())
print('输出 ZIP SHA-256:', hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())
print('在 Kaggle Output 下载 footballmind-qwen-output.zip，并保存本次 Notebook 版本。')
from IPython.display import display, FileLink
import os
os.chdir('/kaggle/working')
display(FileLink(ARCHIVE.name))
'''
    for i, source in enumerate((bootstrap, execution, download)):
        compile(source, 'repair-cell-' + str(i), 'exec')
    cells = []
    def cell(kind, source):
        entry = {'cell_type': kind, 'id': digest(source.encode())[:12], 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)
    cell('markdown', '''# FootballMind T4 显存修复 v2

继续使用原正文 Dataset，不必重新上传。当前 session 恢复优先使用配套 TXT 的单元代码，以保留原输出。

此 Notebook 适合导入或重新开始：Settings 选择 GPU 并启用 Internet，添加原 Dataset，Run All。已有相同输出文件会校验后复用；新 session 的旧 working 文件可能已丢失。

修复只将注意力 query 分块，每块仍读取完整 K/V，并使用原全局因果位置。正文、模型权重、float32、池化与归一化均保持原合同。没有缩短正文，也没有分段取均值。仍先执行8篇CPU对照。
''')
    cell('code', '%pip install -q "transformers==5.8.0" "huggingface_hub==1.14.0" "safetensors>=0.4.3"\n')
    for source in (bootstrap, execution, download):
        cell('code', source)
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    import nbformat
    nbformat.validate(nb)
    (OUTPUT / 'FootballMind_Qwen_T4_Fix_v2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    # This file contains only executable Python. Paste it as one NEW cell in the
    # existing failed notebook; it intentionally performs no package reinstall.
    (OUTPUT / 'paste_this_cell_to_resume.txt').write_text(bootstrap + '\n' + execution + '\n' + download, encoding='utf-8')
    (OUTPUT / 'README.txt').write_text('''T4 显存修复：最短恢复方法

1. 保留当前 Kaggle Notebook/session、原 Input Dataset 和 working 输出，不删除或重新建立实验。
2. 打开 paste_this_cell_to_resume.txt，完整复制到当前 Notebook 的一个新代码单元，运行该单元。不要重新运行原来的失败调用单元。
3. 新单元内嵌经SHA核验的修正版，不要求改只读Dataset或重新上传正文。先重做8篇CPU对照，再校验并跳过已完成向量，继续剩余正文。
4. 正常完成后下载 footballmind-qwen-output.zip，放到 E:/FootballMind/incoming/。结果回收器已兼容旧/新版执行元数据混合。

如果当前session已经丢失，可导入 FootballMind_Qwen_T4_Fix_v2.ipynb，添加同一原Dataset，启用GPU和Internet后Run All。这种情况下不能保证旧working向量仍在。

日志中的HF_TOKEN警告不阻止公开模型下载；本次六个模型文件已下载且权重已加载，异常发生在注意力计算。
原runner使用cuda默认设备0；T4两卡没有自动汇总成一个显存池。新版也先在一张T4上修复长文内存峰值，第二张卡不作为本次成功的必要条件。
15446-token正文的16头FP32完整score张量约14.2204GiB；query分块256后，单块同类张量约0.23569GiB。这不是总显存保证，权重/其他临时张量另占空间；新版记录实际峰值供核对。
保持完整K/V、正确全局causal mask、同权重、float32、last-token pooling及L2。不能把每段文章独立embedding再平均当作此次修复。
本地无CUDA；已做数值与协议测试，实际T4峰值和完整执行仍需用户Notebook运行确认。
''', encoding='utf-8')
    manifest = {p.name: digest(p.read_bytes()) for p in sorted(OUTPUT.iterdir()) if p.is_file()}
    (OUTPUT / 'manifest.json').write_text(json.dumps({'job_sha256': job_sha, 'files_sha256': manifest, 'old_dataset_modified': False, 'old_models_modified': False, 'actual_gpu_execution': False}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'files': {p.name: p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()}}))


if __name__ == '__main__':
    main()
