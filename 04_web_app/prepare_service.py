"""Prepare official member-3 source and an identical public BGE model locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = '7ab02bd824d8b8d24f0721e67f1899d3c515c9a5'
MODEL_REVISION = '7999e1d3359715c523056ef9478215996d62a620'


def source(repo):
    repo = repo.resolve()
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Expected member3 commit {COMMIT}, got {actual}')
    target = ROOT / 'runtime' / 'member3'
    target.mkdir(parents=True, exist_ok=True)
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', COMMIT, '03_weaviate_stream_eval'], text=True).splitlines()
    manifest = {}
    for name in paths:
        relative = Path(name).relative_to('03_weaviate_stream_eval')
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = subprocess.check_output(['git', '-C', str(repo), 'show', f'{COMMIT}:{name}'])
        out.write_bytes(payload)
        manifest[str(relative)] = hashlib.sha256(payload).hexdigest()
    (target / 'source-manifest.json').write_text(json.dumps({'commit': COMMIT, 'sha256': manifest}, indent=2), encoding='utf-8')


def model():
    target = ROOT / 'runtime' / 'bge-small-zh-v1.5'
    files = ['config.json', 'modules.json', 'sentence_bert_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'vocab.txt', '1_Pooling/config.json', 'model.safetensors']
    manifest = {}
    for name in files:
        out = target / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            temp = out.with_suffix(out.suffix + '.download')
            url = f'https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/{MODEL_REVISION}/{name}'
            print(f'Downloading {name}', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '15', '--max-time', '600', url, '-o', str(temp)], check=True)
            temp.replace(out)
        manifest[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    (target / 'download-manifest.json').write_text(json.dumps({'repository': 'BAAI/bge-small-zh-v1.5', 'revision': MODEL_REVISION, 'sha256': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--model-only', action='store_true')
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    if not args.model_only:
        repo = args.source_repo or ROOT / 'runtime' / 'member3-source'
        if not repo.exists():
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'dev-3', 'https://github.com/magicman667/douban-rec-system.git', str(repo)], check=True)
        source(repo)
    if not args.source_only:
        model()
