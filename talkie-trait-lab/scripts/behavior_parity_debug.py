"""Diagnose next-token vs candidate_scores disagreement against an unpadded FP32 reference.

Run on an 80 GB GPU: the 13B model is scored in BF16 (batched and one-at-a-time, both scorers) and
then cast to FP32 and scored one sequence at a time without padding.
"""
import argparse
import json
import random
from pathlib import Path
import torch
from trait_lab.models import load
from trait_lab.train import candidate_scores
from behavior_evals.items import jsonl, render
from behavior_evals.score import next_token_scores, argmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True); p.add_argument('--adapter', type=Path)
    p.add_argument('--items', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--eval', action='append', default=None); p.add_argument('--n', type=int, default=48)
    p.add_argument('--no-fp32', action='store_true', help='Use unpadded BF16 next-token scoring as the reference')
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    model, tok = load(a.family, None, a.adapter)
    rng = random.Random(20261005)
    samples = {}
    for name in a.eval or ['arc_easy', 'global_opinions', 'persona']:
        items = rng.sample(jsonl(a.items / f'{name}.jsonl'), a.n)
        for interface in ('bare', 'chat'):
            samples[f'{interface}/{name}'] = ([render(r, interface) for r in items], [r['candidates'] for r in items])
    variants = {}
    for key, (prompts, cands) in samples.items():
        pairs = [(pr, c) for pr, cs in zip(prompts, cands) for c in cs]
        variants[key] = {
            'fast_batched': [s['logprobs'] for s in next_token_scores(model, tok, prompts, cands)],
            'fast_single': [s['logprobs'] for s in next_token_scores(model, tok, prompts, cands, max_batch=1)],
            'slow_batched': regroup(candidate_scores(model, tok, pairs), cands),
            'slow_single': regroup([candidate_scores(model, tok, [x])[0] for x in pairs], cands)}
    if a.no_fp32:
        for v in variants.values():
            v['fp32_single'] = v['fast_single']
    else:
        model = model.float()
        for key, (prompts, cands) in samples.items():
            variants[key]['fp32_single'] = [s['logprobs'] for s in next_token_scores(model, tok, prompts, cands, max_batch=1)]
    report = {}
    for key, v in variants.items():
        truth = v['fp32_single']
        report[key] = {name: {'mean_abs_error_vs_fp32': mean_abs(x, truth),
                              'argmax_agreement_vs_fp32': sum(argmax(a_) == argmax(b) for a_, b in zip(x, truth)) / len(x)}
                       for name, x in v.items() if name != 'fp32_single'}
        report[key]['examples'] = [{n: [round(z, 3) for z in x[i]] for n, x in v.items()} for i in range(3)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {n: r for n, r in v.items() if n != 'examples'} for k, v in report.items()}, indent=1))


def regroup(flat, cands):
    out, k = [], 0
    for cs in cands:
        out.append(flat[k:k + len(cs)]); k += len(cs)
    return out


def mean_abs(x, y):
    d = [abs(p - q) for a, b in zip(x, y) for p, q in zip(a, b)]
    return sum(d) / len(d)


if __name__ == '__main__':
    main()
