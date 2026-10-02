"""Frozen nested demonstration banks; validation-only, paired pole controls."""
import csv
from collections import Counter
from pathlib import Path
import os
import pyarrow.parquet as pq
from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
from .paths import BUNDLE, REPO
from .io import read, sha, write, verify_bundle
from .fewshot_protocol import conditions, render_demos, validate_banks
from .eval import task, TEMPLATES, evaluate_rows

def inputs(config_path=None):
    verify_bundle()
    config = read(config_path or REPO / 'configs/fewshot.json')
    if config['evaluation_split'] != 'validation':
        raise ValueError('This procedure keeps the final test reserved')
    source = pq.read_table(BUNDLE / 'trait.parquet').to_pylist()
    with (BUNDLE / 'fewshot/scenarios.csv').open() as f:
        splits = {r['scenario_id']: r for r in csv.DictReader(f)}
    banks = read(BUNDLE / 'fewshot/banks.json')
    validate_banks(banks, source, splits, config)
    rows = [r | {'group_id': splits[str(r['source_scenario_id'])]['group_id']}
            for r in source if splits[str(r['source_scenario_id'])]['split'] == 'validation']
    assert len(rows) == 3200
    return config, banks, rows

def choose(config, ids=None):
    available = conditions(config)
    if not ids:
        return available
    unknown = set(ids) - {c['id'] for c in available}
    if unknown:
        raise ValueError(f'Unknown condition IDs: {sorted(unknown)}')
    return [c for c in available if c['id'] in ids]

def check_context(tok, rows, selected, banks, reserve=8):
    lengths = []
    for condition in selected:
        demos = render_demos(condition, banks, task())
        maximum = 0
        for row in rows:
            prefix = demos + TEMPLATES['bare'].format(**row)
            prefix_n = len(tok.encode(prefix, add_special_tokens=False))
            for answer in row['choices']:
                n = prefix_n + len(tok.encode(' ' + str(answer), add_special_tokens=False))
                maximum = max(maximum, n)
        if maximum + reserve > 4096:
            raise ValueError(f"Overlength condition {condition['id']}: {maximum} + {reserve}")
        lengths.append({'id': condition['id'], 'max_candidate_tokens': maximum})
    return lengths

def check(family, ids=None, config_path=None):
    config, banks, rows = inputs(config_path)
    tok = TalkieTokenizer.from_pretrained(BUNDLE / 'interfaces' / family)
    return check_context(tok, rows, choose(config, ids), banks, config['boundary_buffer_tokens'])

def run(family, output, ids=None, config_path=None, base=None, adapter=None, smoke_per_group=0):
    from .models import load
    from .train import select_trait_subset
    config, banks, rows = inputs(config_path)
    selected = choose(config, ids)
    # Audit the full validation panel for the chosen conditions even for smoke.
    tokenizer = TalkieTokenizer.from_pretrained(BUNDLE / 'interfaces' / family)
    lengths = check_context(tokenizer, rows, selected, banks, config['boundary_buffer_tokens'])
    if smoke_per_group:
        rows = select_trait_subset(rows, smoke_per_group)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    plan = {'family': family, 'conditions': selected, 'config': config,
        'trait_sha256': sha(BUNDLE / 'trait.parquet'), 'banks_sha256': sha(BUNDLE / 'fewshot/banks.json'),
        'splits_sha256': sha(BUNDLE / 'fewshot/scenarios.csv'), 'lengths': lengths,
        'smoke_per_group': smoke_per_group, 'base': str(base) if base else None,
        'adapter_sha256': sha(Path(adapter) / 'adapter.safetensors') if adapter else None,
        'item_ids': [r['id'] for r in rows],
        'code': {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / 'src').rglob('*.py'))}}
    planfile = root / 'plan.json'
    if planfile.exists() and read(planfile) != plan:
        raise ValueError('Existing few-shot plan differs; use a new output directory')
    write(planfile, plan)
    model, tok = load(family, base, adapter)
    for condition in selected:
        target = root / condition['id']
        metadata = {'family': family, 'condition': condition, 'plan_sha256': sha(planfile)}
        if (target / 'summary.json').exists():
            summary = read(target / 'summary.json')
            if summary['metadata'] != metadata or sha(target / 'results.jsonl') != summary['results_sha256']:
                raise ValueError(f'Changed completed condition: {target}')
            continue
        if target.exists():
            raise RuntimeError(f'Interrupted attempt retained at {target}; use a new output directory for retry')
        evaluate_rows(model, tok, rows, target, 'bare', metadata,
            prefix=render_demos(condition, banks, task()), chunk_rows=config['chunk_rows'],
            smoke=bool(smoke_per_group))
    write(root / 'complete.json', {'status': 'complete', 'plan_sha256': sha(planfile),
        'conditions': len(selected), 'pairs': len(rows) * len(selected), 'job_id': os.environ.get('SLURM_JOB_ID')})
