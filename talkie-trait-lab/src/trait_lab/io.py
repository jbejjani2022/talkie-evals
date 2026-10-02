import hashlib
import json
import os
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        import gzip
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def verify_bundle():
    from .paths import BUNDLE
    manifest = read(BUNDLE / 'manifest.json')
    for name, item in manifest['files'].items():
        path = BUNDLE / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Changed bundled file: {name}')
    return manifest
