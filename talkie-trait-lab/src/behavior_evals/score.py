"""Next-token candidate scoring for one-token candidates, with parity and TRAIT-reference checks."""
import json
import math
import os
import random
import time
from pathlib import Path
import torch
from trait_lab.io import read, sha, write
from .items import EVALS, jsonl, render


def candidate_ids(tokenizer, candidates):
    ids = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    if any(len(x) != 1 for x in ids):
        raise ValueError(f'Candidates must be single tokens: {candidates} -> {ids}')
    return [x[0] for x in ids]


@torch.inference_mode()
def next_token_scores(model, tokenizer, prompts, candidates, max_tokens=16384, max_batch=64):
    """Log-probabilities of each prompt's one-token candidates, plus the full-vocabulary top token.

    Prompts are length-sorted and left-padded; position IDs follow the attention mask, so padding
    does not shift positions. Only the final position's logits are computed.
    """
    device = next(model.parameters()).device
    # The model's own head is a BF16 matmul, which quantizes logits to steps of ~0.125 nats at typical
    # magnitudes and creates exact candidate ties. Project the final hidden state in FP32 instead.
    head = model.lm_head_gain(model.lm_head).float()
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if any(not e or len(e) >= 4096 for e in encoded):
        raise ValueError('Prompt is empty or exceeds the 4096-token context')
    targets = [candidate_ids(tokenizer, c) for c in candidates]
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    results = [None] * len(encoded)
    start = 0
    while start < len(order):
        end = start + 1
        while (end < len(order) and end - start < max_batch
               and len(encoded[order[end]]) * (end - start + 1) <= max_tokens):
            end += 1
        batch = order[start:end]
        width = len(encoded[batch[-1]])
        ids = torch.full((len(batch), width), tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, index in enumerate(batch):
            n = len(encoded[index])
            ids[row, width - n:] = torch.tensor(encoded[index])
            mask[row, width - n:] = 1
        hidden = model._forward_hidden(ids.to(device), None, mask.to(device), None, None, False)[0]
        logits = hidden[:, -1].float() @ head.T
        logp = torch.log_softmax(logits, dim=-1)
        top = logp.max(dim=-1)
        for row, index in enumerate(batch):
            results[index] = {'logprobs': logp[row, targets[index]].tolist(), 'top_id': int(top.indices[row]),
                              'top_logprob': float(top.values[row]), 'prompt_tokens': len(encoded[index])}
        start = end
    return results


def full_answer_scores(model, tokenizer, prompts, candidates):
    """TRAIT's validated scorer (candidate_scores) on whole option texts, with byte-length normalization.

    'logprobs' holds each option's log-probability per UTF-8 byte times the item's mean option length, so
    argmax is TRAIT's minimum-bits-per-byte rule and a softmax gives a length-normalized distribution.
    """
    from trait_lab.train import candidate_scores
    pairs = [(p, c) for p, cs in zip(prompts, candidates) for c in cs]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))  # tighter padding
    flat = [None] * len(pairs)
    for i, value in zip(order, candidate_scores(model, tokenizer, [pairs[i] for i in order])):
        flat[i] = value
    results, k = [], 0
    for cs in candidates:
        total = flat[k:k + len(cs)]
        k += len(cs)
        sizes = [len(c.encode('utf-8')) for c in cs]
        scale = sum(sizes) / len(sizes)
        results.append({'logprobs': [t / n * scale for t, n in zip(total, sizes)], 'total_logprobs': total, 'bytes': sizes})
    return results


STOPS = ('<|endoftext|>', '<|user|>', '<|assistant|>', '<|system|>', '\n\nQuestion:')


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens=80, batch_size=32):
    """Greedy continuations of the exact likelihood prompts, cut at the first turn/question boundary."""
    device = next(model.parameters()).device
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    out = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        width = max(len(encoded[i]) for i in batch)
        ids = torch.full((len(batch), width), tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(batch):
            ids[row, width - len(encoded[i]):] = torch.tensor(encoded[i]); mask[row, width - len(encoded[i]):] = 1
        generated = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=False,
                                   max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        for row, i in enumerate(batch):
            text = tokenizer.decode(generated[row, width:].tolist(), skip_special_tokens=False)
            cut = min([text.index(s) for s in STOPS if s in text] + [len(text)])
            out[i] = {'raw': text, 'response': text[:cut].strip()}
    return out


def run_generation(family, arm, items_path, output_root, interfaces, model_path=None, adapter=None):
    from trait_lab.models import load
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load(family, model_path, adapter)
    model.config.use_cache = True
    items = jsonl(items_path)
    done = {}
    for interface in interfaces:
        output = Path(output_root) / arm / interface / 'generations.jsonl'
        if output.exists():
            continue
        began = time.time()
        results = generate(model, tokenizer, [render(r, interface) for r in items])
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.partial')
        with temporary.open('w') as f:
            for r, g in zip(items, results):
                f.write(json.dumps({'gen_id': r['gen_id'], 'id': r['id'], 'eval': r['eval'], **g}) + '\n')
        temporary.replace(output)
        write(output.with_suffix('.summary.json'), {'rows': len(items), 'items_sha256': sha(items_path), 'arm': arm,
              'interface': interface, 'seconds': round(time.time() - began, 1), 'decoding': 'greedy, 80 new tokens',
              'job_id': os.environ.get('SLURM_JOB_ID')})
        done[interface] = {'rows': len(items), 'seconds': round(time.time() - began, 1)}
        print(json.dumps({interface: done[interface]}), flush=True)
    return done


def score_eval(model, tokenizer, items_path, output, interface, metadata):
    """Score one eval file for one interface; resumable at file granularity."""
    output = Path(output)
    summary_path = output.with_suffix('.summary.json')
    items_sha = sha(items_path)
    if summary_path.exists():
        summary = read(summary_path)
        if summary['items_sha256'] != items_sha or sha(output) != summary['results_sha256']:
            raise ValueError(f'Existing {output} does not match the current items; use a new output root')
        return summary
    items = jsonl(items_path)
    began = time.time()
    scorer = full_answer_scores if Path(items_path).stem.endswith('_text') else next_token_scores
    scores = scorer(model, tokenizer, [render(r, interface) for r in items], [r['candidates'] for r in items])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    with temporary.open('w') as f:
        for item, s in zip(items, scores):
            if not all(math.isfinite(x) for x in s['logprobs']):
                raise ValueError(f'Non-finite score: {item["id"]}')
            f.write(json.dumps({'id': item['id'], **s}) + '\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(output)
    summary = {'status': 'complete', 'interface': interface, 'rows': len(items), 'items_sha256': items_sha,
               'results_sha256': sha(output), 'seconds': round(time.time() - began, 1), 'metadata': metadata}
    write(summary_path, summary)
    return summary


def run(family, arm, items_root, output_root, interfaces, evals, model_path=None, adapter=None):
    from trait_lab.models import load
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load(family, model_path, adapter)
    metadata = {'family': family, 'arm': arm, 'adapter': str(adapter) if adapter else None,
                'adapter_sha256': read(Path(adapter) / 'adapter.json')['sha256'] if adapter else None,
                'job_id': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__,
                'device': torch.cuda.get_device_name(0)}
    done = {}
    for interface in interfaces:
        for name in evals:
            summary = score_eval(model, tokenizer, Path(items_root) / f'{name}.jsonl',
                                 Path(output_root) / arm / interface / f'{name}.jsonl', interface, metadata)
            done[f'{interface}/{name}'] = {'rows': summary['rows'], 'seconds': summary['seconds']}
            print(json.dumps({f'{interface}/{name}': done[f'{interface}/{name}']}), flush=True)
    return done


def validate(family, arm, items_root, output, model_path=None, adapter=None, per_eval=48, trait_rows=256, seed=20261005):
    """GPU checks before full runs: next-token scorer vs canonical candidate_scores, TRAIT references, timing."""
    from trait_lab.models import load
    from trait_lab.paths import BUNDLE
    from trait_lab.train import candidate_scores
    from trait_lab.eval import score_pair
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load(family, model_path, adapter)
    rng = random.Random(seed)
    report = {'family': family, 'arm': arm, 'job_id': os.environ.get('SLURM_JOB_ID'), 'parity': {}, 'timing': {}}
    for name in EVALS:
        items = jsonl(Path(items_root) / f'{name}.jsonl')
        sample = rng.sample(items, min(per_eval, len(items)))
        for interface in ('bare', 'chat'):
            prompts = [render(r, interface) for r in sample]
            fast = next_token_scores(model, tokenizer, prompts, [r['candidates'] for r in sample])
            pairs = [(p, c) for p, r in zip(prompts, sample) for c in r['candidates']]
            slow = candidate_scores(model, tokenizer, pairs)
            errors, same_choice, k = [], 0, 0
            for s, r in zip(fast, sample):
                reference = slow[k:k + len(r['candidates'])]
                k += len(r['candidates'])
                errors += [abs(a - b) for a, b in zip(s['logprobs'], reference)]
                same_choice += argmax(s['logprobs']) == argmax(reference)
            report['parity'][f'{interface}/{name}'] = {'max_abs_error': max(errors), 'mean_abs_error': sum(errors) / len(errors),
                'argmax_agreement': same_choice / len(sample), 'mean_candidate_mass': sum(
                    sum(math.exp(x) for x in s['logprobs']) for s in fast) / len(fast)}
        timed = rng.sample(items, min(512, len(items)))
        began = time.time()
        next_token_scores(model, tokenizer, [render(r, 'bare') for r in timed], [r['candidates'] for r in timed])
        report['timing'][name] = {'items': len(timed), 'seconds': round(time.time() - began, 2)}
    dataset = read(Path(adapter) / 'adapter.json')['dataset'] if adapter else 'tulu'
    reference_path = BUNDLE / 'references' / f'{family}-{dataset}-{1000 if adapter else 0}-bare.jsonl'
    reference = rng.sample(jsonl(reference_path), trait_rows)
    candidates = [(r['model_input'], c) for r in reference for c in r['continuations']]
    actual = candidate_scores(model, tokenizer, candidates)
    errors, agree = [], 0
    for i, r in enumerate(reference):
        value = score_pair(actual[2 * i:2 * i + 2], r['continuations'], int(r['example']['answer_index']))
        agree += value['high_selected'] == r['high_trait_selected']
        errors += [abs(a - b) for a, b in zip(actual[2 * i:2 * i + 2], r['log_probabilities'])]
    errors.sort()
    report['trait_reference'] = {'reference': reference_path.name, 'rows': trait_rows, 'choice_agreement': agree / trait_rows,
        'median_abs_logprob_error': errors[len(errors) // 2], 'max_abs_logprob_error': errors[-1]}
    write(Path(output), report)
    return report


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)
