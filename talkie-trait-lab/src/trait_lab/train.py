"""Portable single-GPU LoRA training with synchronous raw-answer-BPB TRAIT.

No full-model exports: adapters plus restartable optimizer/RNG checkpoints.
Use sbatch on SLURM hosts; a dedicated workstation can run directly.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import yaml
from talkie_base_experiments.modeling_talkie import TalkieForCausalLM
from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
from talkie_base_experiments.training.common import ATOMIC_CHAT_TEMPLATE, ROLE_TOKENS
from talkie_base_experiments.training.lora import adapter_state, load_adapter_state, inject_lora, collate_rows, batch_indices
from .paths import BUNDLE, REPO, require_gpu
ROOT = BUNDLE
FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
TRAIT = BUNDLE / 'trait.parquet'
SUITE = REPO / 'configs/trait.yaml'

def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as h:
        json.dump(data, h, indent=2)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())
    tmp.replace(path)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def loss_sum(model, batch):
    hidden, _, _ = model._forward_hidden(batch['input_ids'], None, batch['attention_mask'], None, None, False)
    labels = batch['labels'][:, 1:]
    keep = labels != -100
    selected = hidden[:, :-1][keep]
    targets = labels[keep]

    def ce(x, y):
        logits = F.linear(x, model.lm_head_gain(model.lm_head)).float()
        return F.cross_entropy(logits, y, reduction='sum')
    losses = [checkpoint(ce, selected[i:i + 256], targets[i:i + 256], use_reentrant=False) for i in range(0, len(targets), 256)]
    return (torch.stack(losses).sum(), len(targets))

@torch.inference_mode()
def candidate_scores(model, tokenizer, candidates, padded_tokens=4096):
    """Canonical separate prefix/continuation tokenization, full-logit scoring."""
    device = next(model.parameters()).device
    prepared = []
    for prefix, answer in candidates:
        left = tokenizer.encode(prefix, add_special_tokens=False)
        right = tokenizer.encode(answer, add_special_tokens=False)
        if not left or not right or len(left) + len(right) > 4096:
            raise ValueError('Invalid TRAIT candidate length')
        prepared.append({'input_ids': left + right, 'assistant_masks': [0] * len(left) + [1] * len(right)})
    result = [None] * len(prepared)
    order = list(range(len(prepared)))
    groups = []
    group = []
    width = 0
    for i, row in enumerate(prepared):
        n = len(row['input_ids'])
        if group and (max(width, n) * (len(group) + 1) > padded_tokens or len(group) >= 8):
            groups.append(group)
            group = []
            width = 0
        group.append(i)
        width = max(width, n)
    if group:
        groups.append(group)
    for group in groups:
        indexes = [order[i] for i in group]
        batch = collate_rows([prepared[i] for i in indexes], tokenizer.eos_token_id, device)
        for j, index in enumerate(indexes):
            padding = batch['input_ids'].shape[1] - len(prepared[index]['input_ids'])
            for tensor in batch.values():
                tensor[j] = torch.roll(tensor[j], padding)
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
        labels = batch['labels'][:, 1:]
        keep = labels != -100
        selected = logits[:, :-1][keep]
        targets = labels[keep]
        owners = torch.arange(len(group), device=device)[:, None].expand_as(labels)[keep]
        sums = torch.zeros(len(group), dtype=torch.float64, device=device)
        for start in range(0, len(targets), 256):
            nll = F.cross_entropy(selected[start:start + 256], targets[start:start + 256], reduction='none')
            sums.scatter_add_(0, owners[start:start + 256], nll.double())
        values = sums.cpu().tolist()
        for j, index in enumerate(indexes):
            result[index] = -values[j]
    return result

def select_trait_subset(rows, scenarios_per_group, seed=20260915):
    groups = defaultdict(set)
    for row in rows:
        groups[row['category'], row['polarity']].add(row['source_scenario_id'])
    rng = random.Random(seed)
    chosen = set()
    for key, ids in sorted(groups.items()):
        chosen.update(((*key, i) for i in rng.sample(sorted(ids), scenarios_per_group)))
    return [row for row in rows if (row['category'], row['polarity'], row['source_scenario_id']) in chosen]

def evaluate(model, tokenizer, path, metadata, limit=0, subset_per_group=0):
    if (path / 'summary.json').exists():
        previous = json.loads((path / 'summary.json').read_text())
        assert previous['metadata'] == metadata and previous['limit'] == limit and (previous.get('subset_per_group', 0) == subset_per_group)
        return previous
    path.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(TRAIT).to_pylist()
    if len(rows) != 16000:
        raise ValueError('Expected full 16000-row TRAIT')
    if limit:
        seen = defaultdict(int)
        subset = []
        for row in rows:
            key = (row['category'], row['polarity'])
            if seen[key] < limit:
                subset.append(row)
                seen[key] += 1
        rows = subset
    if subset_per_group:
        if limit:
            raise ValueError('Cannot combine smoke limit and scientific subset')
        rows = select_trait_subset(rows, subset_per_group)
        assert len(rows) == 16 * subset_per_group * 2
    task = yaml.safe_load(SUITE.read_text())['tasks'][0]
    model.eval()
    started = time.monotonic()
    summaries = {}
    for interface in ('bare', 'chat'):
        counters = defaultdict(lambda: [0, 0])
        fine = defaultdict(lambda: [0, 0])
        output = path / f'{interface}.jsonl'
        temp = output.with_suffix('.jsonl.partial')
        with temp.open('w') as h:
            for start in range(0, len(rows), 128):
                chunk = rows[start:start + 128]
                candidates = []
                for row in chunk:
                    prefix = task['likelihood_prompt_templates'][interface].format(**row)
                    assert len(row['choices']) == 2
                    candidates.extend(((prefix, task['choice_continuation_prefix'] + str(answer)) for answer in row['choices']))
                scores = candidate_scores(model, tokenizer, candidates)
                for i, row in enumerate(chunk):
                    likelihoods = scores[2 * i:2 * i + 2]
                    bpb = [-ll / (math.log(2) * len(candidates[2 * i + j][1].encode('utf-8'))) for j, ll in enumerate(likelihoods)]
                    chosen = min(range(2), key=bpb.__getitem__)
                    high = int(chosen == int(row['answer_index']))
                    category = row['category']
                    polarity = row['polarity']
                    for counter, key in [(counters, category), (fine, category + '/' + polarity)]:
                        counter[key][0] += high
                        counter[key][1] += 1
                    h.write(json.dumps({'index': start + i, 'example': row, 'model_input': candidates[2 * i][0], 'continuations': [candidates[2 * i + j][1] for j in range(2)], 'log_probabilities': likelihoods, 'bits_per_byte': bpb, 'selected_index': chosen, 'high_trait_selected': high}) + '\n')
                h.flush()
                if start % 2048 == 0:
                    print(json.dumps({'event': 'trait_progress', 'interface': interface, 'rows': min(start + 128, len(rows)), 'path': str(path)}), flush=True)
            os.fsync(h.fileno())
        temp.replace(output)

        def rates(counter):
            return {k: {'high_selected': v[0], 'rows': v[1], 'high_trait_rate': v[0] / v[1]} for k, v in sorted(counter.items())}
        summaries[interface] = {'rows': len(rows), 'by_category': rates(counters), 'by_category_and_polarity': rates(fine), 'results_sha256': sha(output)}
    result = {'status': 'complete', 'metadata': metadata, 'limit': limit, 'subset_per_group': subset_per_group, 'trait_sha256': sha(TRAIT), 'suite_sha256': sha(SUITE), 'elapsed_seconds': time.monotonic() - started, 'interfaces': summaries}
    write_json(path / 'summary.json', result)
    return result

def parity_check(model, tokenizer):
    from .reference import TransformersBackend
    from types import SimpleNamespace
    EvalModelConfig = SimpleNamespace
    candidates = [('Question: A friend needs help.\n\nAnswer:', ' I would help my friend.'), ('Question: A friend needs help.\n\nAnswer:', ' I would ignore the request.')]
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.config = EvalModelConfig(path='unused', batch_size=2, max_model_len=4096)
    backend.model = model
    backend.tokenizer = tokenizer
    old = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    model.eval()
    reference = backend.loglikelihood([x for x, y in candidates], [y for x, y in candidates])
    actual = candidate_scores(model, tokenizer, candidates)
    tokenizer.padding_side = old
    errors = [abs(a - r.log_probability) for a, r in zip(actual, reference)]
    assert max(errors) < 0.001, errors
    return {'max_log_probability_error': max(errors), 'reference': [r.log_probability for r in reference], 'actual': actual}

def save_checkpoint(path, model, optimizer, state, args):
    if path.exists():
        raise FileExistsError(path)
    stage = path.with_name(path.name + '.partial')
    stage.mkdir(parents=True)
    torch.save({'adapter': adapter_state(model), 'optimizer': optimizer.state_dict(), 'state': state, 'args': vars(args) | {'data': str(args.data), 'output': str(args.output), 'model': str(args.model) if args.model else None, 'resume': str(args.resume) if args.resume else None}, 'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'python': random.getstate(), 'numpy': np.random.get_state()}}, stage / 'checkpoint.pt')
    with (stage / 'checkpoint.pt').open('rb') as h:
        os.fsync(h.fileno())
    write_json(stage / 'metadata.json', state)
    write_json(stage / 'validation.json', {'sha256': sha(stage / 'checkpoint.pt'), 'bytes': (stage / 'checkpoint.pt').stat().st_size, 'status': 'complete'})
    stage.rename(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=['vintage', 'web'], required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=float, default=32)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--batch-tokens', type=int, default=4096)
    p.add_argument('--update-tokens', type=int, default=16384)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--trait-limit', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.set_defaults(async_eval_dir=None)
    p.add_argument('--endpoint-only', action='store_true')
    p.add_argument('--model', type=Path)
    args = p.parse_args()
    require_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU')
    if args.trait_limit and (not args.smoke_steps):
        raise ValueError('Partial TRAIT is smoke-only')
    torch.set_num_threads(4)
    seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    assert sha(TRAIT) == manifest['source_sha256']['trait']
    datafile = args.data / f'{args.family}-{args.dataset}.parquet'
    assert sha(datafile) == manifest['files'][datafile.name]['sha256']
    rows = pq.read_table(datafile).to_pylist()
    total = sum((r['loss_tokens'] for r in rows))
    batches = batch_indices(rows, args.batch_tokens)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'run.json').exists() and (not args.resume):
        raise FileExistsError('Existing run; supply --resume')
    base = str(args.model) if args.model else manifest['models'][args.family]
    from .models import validate_identity
    validate_identity(base, args.family)
    model = TalkieForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    tokenizer = TalkieTokenizer.from_pretrained(base)
    tokenizer.add_special_tokens({'additional_special_tokens': list(ROLE_TOKENS)})
    tokenizer.chat_template = ATOMIC_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    assert len(tokenizer) == model.config.vocab_size
    probe = torch.tensor([rows[0]['input_ids'][:64]], device='cuda')
    with torch.inference_mode():
        before = model(probe, use_cache=False).logits.clone()
    targets = inject_lora(model, args.rank, args.alpha)
    assert len(targets) == 7 * model.config.n_layer
    model.eval()
    with torch.inference_mode():
        after = model(probe, use_cache=False).logits
    assert torch.equal(before, after), 'Zero adapters changed logits'
    del before, after
    for block in model.blocks:
        original = block.forward

        def wrapped(*a, _forward=original, **kw):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *a, use_reentrant=False, **kw)
            return _forward(*a, **kw)
        block.forward = wrapped
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(('lora_' in n for n, p in model.named_parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    state = {'step': 0, 'cursor': 0, 'loss_tokens': 0, 'input_tokens': 0, 'completed_fractions': [], 'base': base, 'family': args.family, 'dataset': args.dataset, 'total_loss_tokens': total, 'data_manifest_sha256': sha(args.data / 'manifest.json')}
    initial_state = json.loads(json.dumps(state))
    if args.resume:
        validation = json.loads((args.resume / 'validation.json').read_text())
        assert sha(args.resume / 'checkpoint.pt') == validation['sha256']
        previous = torch.load(args.resume / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for key in ('base', 'family', 'dataset', 'total_loss_tokens', 'data_manifest_sha256'):
            assert previous['state'][key] == state[key], key
        for key in ('rank', 'alpha', 'lr', 'seed', 'batch_tokens', 'update_tokens', 'smoke_steps', 'trait_limit'):
            assert previous['args'][key] == vars(args)[key], key
        load_adapter_state(model, previous['adapter'])
        optimizer.load_state_dict(previous['optimizer'])
        state = previous['state']
        torch.set_rng_state(previous['rng']['torch'])
        torch.cuda.set_rng_state_all(previous['rng']['cuda'])
        random.setstate(previous['rng']['python'])
        np.random.set_state(previous['rng']['numpy'])
    run = {'status': 'running', 'job_id': os.environ.get('SLURM_JOB_ID'), 'args': vars(args) | {'data': str(args.data), 'output': str(args.output), 'resume': str(args.resume) if args.resume else None, 'model': str(args.model) if args.model else None}, 'trainable_parameters': sum((p.numel() for p in params)), 'targets': targets, 'base': base, 'data_manifest_sha256': state['data_manifest_sha256'], 'total_loss_tokens': total, 'rows': len(rows), 'torch': torch.__version__, 'deterministic_algorithms': True, 'zero_adapter_exact': True, 'likelihood_parity': parity_check(model, tokenizer), 'code_sha256': {str(path): sha(path) for path in [Path(__file__), Path(__import__('talkie_base_experiments.training.lora', fromlist=['x']).__file__)]}}
    write_json(args.output / 'run.json', run)
    milestones = [1.0] if args.smoke_steps else list(FRACTIONS)
    if args.resume and state['completed_fractions'] and (not args.endpoint_only):
        fraction = state['completed_fractions'][-1]
        evaluate(model, tokenizer, args.output / 'trait' / f'fraction-{round(fraction * 1000):03d}', dict(state), args.trait_limit)
    if not args.smoke_steps and 0.0 not in state['completed_fractions']:
        evaluate(model, tokenizer, args.output / 'trait' / 'fraction-000', dict(state))
        state['completed_fractions'].append(0.0)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    accum = 0
    accum_loss = 0.0
    log = args.output / 'training.jsonl'
    for cursor in range(state['cursor'], len(batches)):
        model.train()
        selected = [rows[i] for i in batches[cursor]]
        batch = collate_rows(selected, tokenizer.eos_token_id, 'cuda')
        loss, count = loss_sum(model, batch)
        assert count == sum((r['loss_tokens'] for r in selected))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        (loss / args.update_tokens).backward()
        accum += count
        accum_loss += float(loss.detach())
        state['cursor'] = cursor + 1
        state['loss_tokens'] += count
        state['input_tokens'] += sum((len(r['input_ids']) for r in selected))
        pending = [f for f in milestones if f not in state['completed_fractions']]
        next_fraction = pending[0] if pending else 1.0
        crossed = state['loss_tokens'] >= math.ceil(total * next_fraction)
        last = cursor == len(batches) - 1
        if accum < args.update_tokens and (not crossed) and (not last):
            continue
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.mul_(args.update_tokens / accum)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        midpoint = (state['loss_tokens'] - accum / 2) / total
        factor = min(1.0, max(midpoint, 1 / total) / 0.03) if midpoint < 0.03 else max(0.0, (1.0 - midpoint) / 0.97)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * factor
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state['step'] += 1
        record = dict(state, job_id=os.environ.get('SLURM_JOB_ID'), loss=accum_loss / accum, gradient_norm=float(norm), lr=optimizer.param_groups[0]['lr'], update_loss_tokens=accum, elapsed_seconds=time.monotonic() - started, peak_memory_gb=torch.cuda.max_memory_allocated() / 1000000000.0)
        with log.open('a') as h:
            h.write(json.dumps(record) + '\n')
            h.flush()
            os.fsync(h.fileno())
        print(json.dumps({'event': 'update', **record}), flush=True)
        accum = 0
        accum_loss = 0.0
        smoke_done = args.smoke_steps and state['step'] >= args.smoke_steps
        if crossed or last or smoke_done:
            reached = [f for f in pending if state['loss_tokens'] >= math.ceil(total * f)]
            if smoke_done:
                reached = [state['loss_tokens'] / total]
            if len(reached) > 1:
                raise RuntimeError('Update skipped multiple requested milestones; reduce batch size')
            fraction = reached[0]
            check = args.output / 'checkpoints' / f'step-{state['step']:06d}'
            state['completed_fractions'].extend(reached)
            save_checkpoint(check, model, optimizer, dict(state), args)
            saved = torch.load(check / 'checkpoint.pt', map_location='cpu', weights_only=False)['adapter']
            assert all((torch.equal(v, saved[k]) for k, v in adapter_state(model).items()))
            if not args.endpoint_only or last or smoke_done:
                evaluate(model, tokenizer, args.output / 'trait' / f'fraction-{round(fraction * 1000):03d}', dict(state), args.trait_limit)
            write_json(args.output / 'progress.json', state)
            if smoke_done:
                break
    if not args.smoke_steps:
        assert state['loss_tokens'] == total and state['cursor'] == len(batches)
        assert set(state['completed_fractions']) == set(FRACTIONS)
    assert any((torch.count_nonzero(v) for k, v in adapter_state(model).items() if k.endswith('lora_B')))
    run.update(status='complete', evaluation_mode='synchronous', state=state, peak_memory_gb=torch.cuda.max_memory_allocated() / 1000000000.0, elapsed_seconds=time.monotonic() - started)
    from .adapters import export_run
    export_run(args.output, model, args.family, args.rank, args.alpha, state, base)
    write_json(args.output / 'run.json', run)
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'state': state}), flush=True)
if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import sys
        if '--output' in sys.argv:
            status_path = Path(sys.argv[sys.argv.index('--output') + 1]) / 'run.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get('job_id') == os.environ.get('SLURM_JOB_ID'):
                    status.update(status='failed', error=repr(error))
                    write_json(status_path, status)
        raise
