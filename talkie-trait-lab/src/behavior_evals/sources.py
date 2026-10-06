"""Pinned downloads for the behavior evals, verified file by file against configs/behavior/sources.json."""
import urllib.request
from pathlib import Path
from trait_lab.io import read, sha
from trait_lab.paths import REPO

SOURCES = REPO / 'configs/behavior/sources.json'


def fetch(destination):
    """Download every pinned file once; refuse any byte that differs from the recorded SHA-256."""
    destination = Path(destination)
    for name, source in read(SOURCES).items():
        for path, digest in source['files'].items():
            target = destination / name / path
            if target.exists() and sha(target) == digest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            url = source['url'].format(repo=source['repo'], revision=source['revision'], path=path)
            urllib.request.urlretrieve(url, temporary)
            if sha(temporary) != digest:
                raise ValueError(f'{name}/{path}: downloaded hash differs from the pinned revision')
            temporary.replace(target)
    return destination
