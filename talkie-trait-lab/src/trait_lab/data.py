"""Pinned data download, canonical TRAIT rendering, and frozen SFT selection."""
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
from talkie_base_experiments.training.common import _tokenize_messages_with_assistant_mask
from .paths import BUNDLE, ROOT, model_path
from .io import read, write, sha, verify_bundle

def download(which):
    manifest = verify_bundle()
    if which in ('trait', 'all'):
        rev = manifest['trait_revision']
        destination = ROOT / 'downloads/TRAIT.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.partial')
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/pull-ups/TRAIT/{rev}/TRAIT.json', temp)
        if sha(temp) != manifest['trait_raw_sha256']:
            raise ValueError('Downloaded TRAIT hash differs from original campaign')
        temp.replace(destination)
        rows = materialize_trait(read(destination))
        original = pq.read_table(BUNDLE / 'trait.parquet').to_pylist()
        fields = ('id', 'prompt', 'choices', 'answer_index', 'category', 'polarity', 'source_scenario_id', 'source_pair')
        assert [{k:r[k] for k in fields} for r in rows] == [{k:r[k] for k in fields} for r in original]
        pq.write_table(pa.Table.from_pylist(rows), ROOT / 'downloads/trait.parquet')
    if which in ('tulu', 'all'):
        snapshot_download(repo_id='allenai/tulu-3-sft-mixture', repo_type='dataset',
            revision=manifest['tulu_revision'], local_dir=ROOT / 'downloads/tulu3',
            allow_patterns=['data/*.parquet', 'README.md', '*LICENSE*'])

def materialize_trait(source, seed=20260831):
    result = []
    for r in source:
        for pair in (1, 2):
            prompt = f"{r['situation']}\n\n{r['query']}"
            choices = [r[f'response_high{pair}'], r[f'response_low{pair}']]
            order = [0, 1]
            payload = json.dumps([seed, prompt, choices], ensure_ascii=False).encode()
            random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')).shuffle(order)
            result.append(dict(id=f"trait-{r['idx']}-{pair}", prompt=prompt,
                choices=[choices[i] for i in order], answer_index=order.index(0), category=r['personality'],
                polarity=r['split'], source_scenario_id=str(r['idx']), source_pair=pair, measure='high_trait_choice'))
    if len(result) != 16000 or len({r['id'] for r in result}) != len(result):
        raise ValueError('Expected 16000 unique paired-choice rows')
    return result

def message_sha(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def prepare_data(families, datasets, base_overrides=None):
    verify_bundle()
    selections = read(BUNDLE / 'selections.json.gz')
    known = {v['dataset'] for v in selections.values()}
    if any(d not in known and d != 'all' for d in datasets):
        raise ValueError('Unknown dataset slug; use list-data')
    keys = [k for k, v in selections.items() if v['family'] in families and
            ('all' in datasets or v['dataset'] in datasets)]
    if not keys:
        raise ValueError('No matching dataset; use list-data')
    # Some hard-coded examples repeat the same ID with identical text upstream.
    # Retain frozen occurrences/order and validate content, rather than assuming
    # every raw source ID is unique.
    needed = {r['id'] for k in keys if selections[k]['dataset'] != 'vintage' for r in selections[k]['rows']}
    messages = {str(r['id']): r for r in pq.read_table(BUNDLE / 'vintage-sft.parquet').to_pylist()}
    if needed:
        files = sorted((ROOT / 'downloads/tulu3/data').glob('*.parquet'))
        if not files:
            raise FileNotFoundError('Run trait-lab download-data tulu first')
        found = set()
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=2048, columns=['id', 'source', 'messages']):
                for row in batch.to_pylist():
                    key = str(row['id'])
                    if key in needed:
                        if key in found:
                            if messages[key]['source'] != row['source'] or message_sha(messages[key]['messages']) != message_sha(row['messages']):
                                raise ValueError(f'Conflicting duplicate Tulu ID: {key}')
                            continue
                        found.add(key); messages[key] = row
        if found != needed:
            raise ValueError(f'Missing {len(needed-found)} pinned Tulu rows')
    output = ROOT / 'prepared'; output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = read(manifest_path) if manifest_path.exists() else {
        'models': {}, 'files': {}, 'source_sha256': {'trait': sha(BUNDLE / 'trait.parquet')},
        'bundle_manifest_sha256': sha(BUNDLE / 'manifest.json'), 'actual': {}}
    if manifest['bundle_manifest_sha256'] != sha(BUNDLE / 'manifest.json'):
        raise ValueError('Prepared data comes from a different bundle')
    for family in families:
        base = str((base_overrides or {}).get(family) or model_path(family))
        if family in manifest['models'] and manifest['models'][family] != base:
            raise ValueError('Existing prepared tokenizer path changed; use a new artifact root')
        tok = TalkieTokenizer.from_pretrained(base)
        manifest['models'][family] = base
        for key in keys:
            selection = selections[key]
            if selection['family'] != family:
                continue
            destination = output / f'{key}.parquet'
            if destination.name in manifest['files']:
                assert sha(destination) == manifest['files'][destination.name]['sha256']
                continue
            encoded = []
            for expected in selection['rows']:
                source = messages[expected['id']]
                if message_sha(source['messages']) != expected['messages_sha256']:
                    raise ValueError(f'Changed source text: {key}/{expected["id"]}')
                x = _tokenize_messages_with_assistant_mask(source, tok)
                x.update(id=expected['id'], source=source['source'], loss_tokens=sum(x['assistant_masks'][1:]))
                if hashlib.sha256(json.dumps([x['input_ids'], x['assistant_masks']]).encode()).hexdigest() != expected['tokens_sha256']:
                    raise ValueError(f'Token/mask reproduction failed: {key}/{expected["id"]}')
                assert len(x['input_ids']) == expected['input_tokens'] <= 4096
                assert x['loss_tokens'] == expected['loss_tokens'] > 0
                encoded.append(x)
            temporary = destination.with_suffix('.parquet.partial')
            pq.write_table(pa.Table.from_pylist(encoded), temporary); temporary.replace(destination)
            manifest['files'][destination.name] = {'bytes': destination.stat().st_size, 'sha256': sha(destination)}
            manifest['actual'][key] = {k: selection[k] for k in ('input_tokens', 'loss_tokens')}
            manifest['actual'][key]['rows'] = len(encoded)
            write(manifest_path, manifest)
    return output

def prepare_custom(family, name, messages_path, cap=0, seed=20260915, base=None):
    """Explicit new-data path; frozen reproduction selectors remain immutable."""
    import numpy as np
    verify_bundle()
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):
        raise ValueError('Use a lowercase dataset slug')
    frozen = read(BUNDLE / 'selections.json.gz')
    if f'{family}-{name}' in frozen:
        raise ValueError('Use a new name for a new experiment')
    path = Path(messages_path)
    if path.suffix == '.parquet':
        source = pq.read_table(path).to_pylist()
    else:
        source = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not source or len({str(r['id']) for r in source}) != len(source):
        raise ValueError('Need nonempty input and unique IDs')
    if cap < 0:
        raise ValueError('Token cap must be nonnegative')
    base = str(base or model_path(family))
    tok = TalkieTokenizer.from_pretrained(base)
    order = np.random.default_rng(seed).permutation(len(source))
    rows = []; total = 0
    for i in order:
        row = source[int(i)]
        if any(m['role'] not in ('system', 'user', 'assistant') for m in row['messages']):
            raise ValueError(f'Unsupported role in {row["id"]}')
        x = _tokenize_messages_with_assistant_mask(row, tok)
        x.update(id=str(row['id']), source=row.get('source', 'custom'), loss_tokens=sum(x['assistant_masks'][1:]))
        if len(x['input_ids']) > 4096 or x['loss_tokens'] < 1:
            raise ValueError(f'Invalid length/mask in {row["id"]}; no silent truncation')
        if cap and total + x['loss_tokens'] > cap:
            break
        rows.append(x); total += x['loss_tokens']
    if not rows:
        raise ValueError('Token budget does not fit the first shuffled example')
    output = ROOT / 'prepared'; output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = read(manifest_path) if manifest_path.exists() else {
        'models': {}, 'files': {}, 'source_sha256': {'trait': sha(BUNDLE / 'trait.parquet')},
        'bundle_manifest_sha256': sha(BUNDLE / 'manifest.json'), 'actual': {}}
    if manifest['bundle_manifest_sha256'] != sha(BUNDLE / 'manifest.json'):
        raise ValueError('Existing data comes from a different bundle')
    if family in manifest['models'] and manifest['models'][family] != base:
        raise ValueError('Prepared tokenizer changed; use a new artifact root')
    filename = f'{family}-{name}.parquet'; target = output / filename
    if target.exists() or filename in manifest['files']:
        raise FileExistsError('Use a new dataset name or artifact root')
    temp = target.with_suffix('.parquet.partial')
    pq.write_table(pa.Table.from_pylist(rows), temp); temp.replace(target)
    manifest['models'][family] = base
    manifest['files'][filename] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
    manifest['actual'][f'{family}-{name}'] = {'rows': len(rows), 'loss_tokens': total,
        'input_tokens': sum(len(r['input_ids']) for r in rows), 'source_sha256': sha(path),
        'seed': seed, 'cap': cap, 'selected_ids': [r['id'] for r in rows], 'design': 'new custom dataset'}
    write(manifest_path, manifest)
    return target
