"""Download a pinned public Qwen embedding snapshot to the E-drive model cache."""
from pathlib import Path
import os,json,hashlib,shutil,time

def main():
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_ETAG_TIMEOUT']='30'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
    from huggingface_hub import HfApi,hf_hub_download
    repo='Qwen/Qwen3-Embedding-0.6B'
    revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
    root=Path('E:/FootballMind/models');cache=root/'huggingface'
    local=Path('C:/Users/windows/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B')
    target=cache/local.name
    # Reuse only existing nonempty files from exactly this revision. Leave C: intact.
    for sub in [Path('snapshots')/revision,Path('blobs')]:
        if (local/sub).exists():
            for p in (local/sub).rglob('*'):
                if p.is_file():
                    q=target/p.relative_to(local)
                    if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    info=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
    assert info.sha==revision
    wanted=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors','README.md']
    by={s.rfilename:s for s in info.siblings};log=[]
    print(json.dumps({'stage':'download_start','repository':repo,'revision':revision,'weight_bytes':by['model.safetensors'].size}),flush=True)
    for name in wanted:
        p=Path(hf_hub_download(repo,name,revision=revision,cache_dir=cache,token=False))
        digest=hashlib.sha256(p.read_bytes()).hexdigest();lfs=by[name].lfs
        expected=lfs.sha256 if lfs else None
        if expected:assert digest==expected,(name,digest,expected)
        log.append(dict(file=name,path=str(p),bytes=p.stat().st_size,sha256=digest,repository_lfs_sha256=expected))
        print(json.dumps({'stage':'file_ready','file':name,'bytes':p.stat().st_size}),flush=True)
    receipt=dict(repository=repo,revision=revision,model_path=str(target/'snapshots'/revision),files=log,
        downloaded_only_public_model_files=True,training_data_uploaded=False,license='Apache-2.0')
    dest=root/'qwen3-embedding-0.6b-pinned.json';dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'complete','receipt':str(dest)}),flush=True)

if __name__=='__main__':main()
