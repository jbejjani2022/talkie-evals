"""Exact raw-answer conditional BPB, full raw records and per-trait summaries."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import pyarrow.parquet as pq
from .io import write, sha, read, verify_bundle
from .paths import BUNDLE

TEMPLATES = {'bare': 'Question: {prompt}\n\nAnswer:',
             'chat': '<|user|>\n{prompt}\n<|assistant|>\nAnswer:'}

def task():
    return {'likelihood_prompt_templates': TEMPLATES, 'choice_continuation_prefix': ' '}

def score_pair(scores, answers, high_index):
    from .fewshot_protocol import score_pair as original
    return original(scores, answers, high_index)

def evaluate_rows(model, tokenizer, rows, output, interface, metadata, prefix='', chunk_rows=128, smoke=False):
    from .train import candidate_scores
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    temp = output / 'results.jsonl.partial'
    count = defaultdict(lambda: [0, 0, 0.0])
    with temp.open('w') as handle:
        for start in range(0, len(rows), chunk_rows):
            chunk = rows[start:start+chunk_rows]; candidates = []
            for row in chunk:
                prompt = prefix + TEMPLATES[interface].format(**row)
                candidates.extend((prompt, ' ' + str(a)) for a in row['choices'])
            scores = candidate_scores(model, tokenizer, candidates)
            for i, row in enumerate(chunk):
                prompt = candidates[2*i][0]; answers = [candidates[2*i+j][1] for j in (0, 1)]
                value = score_pair(scores[2*i:2*i+2], answers, int(row['answer_index']))
                token_ids = [tokenizer.encode(a, add_special_tokens=False) for a in answers]
                prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
                handle.write(json.dumps({'example': row, 'model_input': prompt, 'continuations': answers,
                    'prefix_token_ids': prefix_ids, 'answer_token_ids': token_ids, **value}) + '\n')
                for key in (row['category'], row['category'] + '/' + row['polarity']):
                    count[key][0] += value['high_selected']; count[key][1] += 1; count[key][2] += value['margin']
            handle.flush(); os.fsync(handle.fileno())
    temp.replace(output / 'results.jsonl')
    summary = {'status': 'complete', 'interface': interface, 'rows': len(rows), 'smoke_only': smoke,
        'metadata': metadata, 'results_sha256': sha(output / 'results.jsonl'),
        'scores': {k: {'high_selected': h, 'rows': n, 'high_trait_rate': h/n, 'mean_margin_bpb': margin/n}
                   for k, (h, n, margin) in count.items()}}
    write(output / 'summary.json', summary)
    return summary

def run(family, output, interface='bare', model_path=None, adapter=None, smoke_limit=0):
    from .models import load
    verify_bundle()
    rows = pq.read_table(BUNDLE / 'trait.parquet').to_pylist()
    if smoke_limit:
        from .train import select_trait_subset
        rows = select_trait_subset(rows, smoke_limit)
    model, tok = load(family, model_path, adapter)
    metadata = {'family': family, 'base': str(model_path) if model_path else None,
        'adapter': str(adapter) if adapter else None, 'trait_sha256': sha(BUNDLE / 'trait.parquet'),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'torch': __import__('torch').__version__}
    return evaluate_rows(model, tok, rows, output, interface, metadata, smoke=bool(smoke_limit))
