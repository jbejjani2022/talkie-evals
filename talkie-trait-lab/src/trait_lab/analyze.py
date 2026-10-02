"""Per-trait effects and paired scenario-group bootstrap intervals."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import read, sha, write

def records(root):
    root = Path(root)
    summary = read(root / 'summary.json')
    if sha(root / 'results.jsonl') != summary['results_sha256']:
        raise ValueError('Result checksum mismatch')
    with (root / 'results.jsonl').open() as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != summary['rows'] or len({r['example']['id'] for r in rows}) != len(rows):
        raise ValueError('Row count or uniqueness mismatch')
    return {r['example']['id']: r for r in rows}

def contrast(left, right, output, replicates=2000, seed=20260925):
    """Compute left-minus-right using matched scenarios, paired answer rows."""
    if read(Path(left) / 'summary.json')['interface'] != read(Path(right) / 'summary.json')['interface']:
        raise ValueError('Bare and chat evaluations must be compared separately')
    if replicates < 1:
        raise ValueError('Need at least one bootstrap replicate')
    a, b = records(left), records(right)
    if a.keys() != b.keys():
        raise ValueError('Comparison panels differ')
    strata = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
    flips = defaultdict(lambda: [0, 0])
    for key, r in a.items():
        s = b[key]; x = r['example']; y = s['example']
        for field in ('prompt', 'choices', 'answer_index', 'category', 'polarity', 'source_scenario_id'):
            if x[field] != y[field]:
                raise ValueError(f'Unmatched input: {key}/{field}')
        trait, polarity = x['category'], x['polarity']
        group = x.get('group_id', str(x['source_scenario_id']))
        high_a = r.get('high_selected', r.get('high_trait_selected'))
        high_b = s.get('high_selected', s.get('high_trait_selected'))
        cell = strata[trait, polarity][group]
        cell[0] += high_a - high_b; cell[1] += 1
        hi = int(x['answer_index']); lo = 1-hi
        margin_a = r['bits_per_byte'][lo] - r['bits_per_byte'][hi]
        margin_b = s['bits_per_byte'][lo] - s['bits_per_byte'][hi]
        cell[2] += margin_a - margin_b
        flips[trait][0] += high_a == 1 and high_b == 0
        flips[trait][1] += high_a == 0 and high_b == 1
    rng = np.random.default_rng(seed); results = []
    for trait in sorted({k[0] for k in strata}):
        numerator = denominator = margin = 0
        draws_num = np.zeros(replicates); draws_den = np.zeros(replicates)
        for (t, polarity), groups in sorted(strata.items()):
            if t != trait: continue
            values = np.asarray(list(groups.values()), dtype=float)
            numerator += values[:,0].sum(); denominator += values[:,1].sum(); margin += values[:,2].sum()
            draws = rng.multinomial(len(values), np.full(len(values), 1/len(values)), size=replicates)
            draws_num += draws @ values[:,0]; draws_den += draws @ values[:,1]
        lo, hi = np.quantile(100*draws_num/draws_den, [0.025, 0.975])
        results.append({'trait': trait, 'rows': denominator, 'delta_pp': 100*numerator/denominator,
            'ci_low_pp': float(lo), 'ci_high_pp': float(hi), 'mean_margin_delta_bpb': margin/denominator,
            'low_to_high': flips[trait][0], 'high_to_low': flips[trait][1]})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    write(output.with_suffix('.provenance.json'), {'left': str(left), 'right': str(right),
        'left_sha256': sha(Path(left)/'results.jsonl'), 'right_sha256': sha(Path(right)/'results.jsonl'),
        'replicates': replicates, 'seed': seed, 'bootstrap': 'paired groups stratified by trait and source polarity',
        'intervals': 'unadjusted marginal percentile intervals conditional on fixed checkpoints/prompts'})
    return results
