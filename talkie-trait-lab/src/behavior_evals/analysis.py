"""Tidy result tables for all arms, interfaces and evals (numpy only; run locally on rsynced scores).

SFT effects are paired: the same resampled items/groups are used for the base and SFT arm, so the
interval is for the within-family SFT - base difference. Intervals are percentile bootstrap intervals
conditional on one training seed per arm and on the fixed item set.
"""
import csv
import json
from pathlib import Path
import numpy as np
from . import metrics
from .items import degenerate, jsonl

ARMS = ['vintage-base', 'vintage-tulu', 'vintage-vsft', 'web-base', 'web-tulu', 'web-vsft']
SFT = {'tulu': 'Tulu 3 SFT', 'vsft': 'Vintage SFT'}
INTERFACES = ['bare', 'chat']


def family_of(arm):
    return arm.split('-')[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Results:
    """Loads every arm's scores for one eval/interface, optionally restricted by an item filter."""

    def __init__(self, items_root, scores_root, keep=None):
        self.items_root, self.scores_root, self.keep = Path(items_root), Path(scores_root), keep

    def load(self, name, interface, arm):
        items, scores = metrics.load(self.items_root / f'{name}.jsonl', self.scores_root / arm / interface / f'{name}.jsonl')
        # Fragment-stem AI-risk items are dropped in both scoring modes so the modes cover the same questions.
        pairs = [(r, s) for r, s in zip(items, scores) if not degenerate(r) and (self.keep is None or self.keep(name, r))]
        return [r for r, _ in pairs], [s for _, s in pairs]

    def arms(self, interface):
        return [a for a in ARMS if (self.scores_root / a / interface).exists()]


def paired_rows(draws, point, arms, base, extra):
    """Rows for each arm's value and for every SFT arm minus its family base."""
    rows = []
    for j, arm in enumerate(arms):
        lo, hi = metrics.interval(draws[:, j])
        rows.append({**extra, 'arm': arm, 'value': point[j], 'ci_low': lo, 'ci_high': hi})
    deltas = []
    for j, arm in enumerate(arms):
        if arm == base or family_of(arm) != family_of(base):
            continue
        b = arms.index(base)
        lo, hi = metrics.interval(draws[:, j] - draws[:, b])
        deltas.append({**extra, 'family': family_of(arm), 'sft': arm.split('-')[1], 'delta': point[j] - point[b],
                       'ci_low': lo, 'ci_high': hi})
    return rows, deltas


def persona(results, interface, offsets):
    rows, deltas, extra_rows = [], [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        loaded = {a: results.load('persona', interface, a) for a in arms}
        items = loaded[arms[0]][0]
        choices = {a: metrics.persona_choices(*loaded[a], offsets[interface, a]) for a in arms}
        behaviors = np.array([r['subset'] for r in items])
        for behavior in sorted(set(behaviors)):
            idx = behaviors == behavior
            strata = choices[arms[0]]['polarity'][idx]
            if len(set(strata)) < 2:
                continue
            values = np.stack([choices[a]['soft'][idx] for a in arms])
            point, draws = metrics.stratified_mean(values, strata)
            extra = {'interface': interface, 'behavior': behavior, 'category': metrics.persona_category(behavior),
                     'n': int(idx.sum())}
            r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
            rows += r; deltas += d
            for a in arms:
                c = choices[a]
                extra_rows.append({**extra, 'arm': a, 'raw_match': 100 * c['raw'][idx].mean(),
                    'calibrated_balanced': 100 * metrics.stratified_mean(c['calibrated'][idx], strata, reps=1)[0][0],
                    'yes_rate': 100 * c['yes'][idx].mean()})
    return rows, deltas, extra_rows


def sycophancy(results, interface):
    rows, deltas = [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        parts = {a: metrics.sycophancy_parts(*results.load('sycophancy', interface, a)) for a in arms}
        groups = parts[arms[0]]['group']
        subsets = np.array([g.split('/')[1] for g in groups])
        for subset in sorted(set(subsets)):
            idx = subsets == subset
            for measure, (obs, null) in {'excess': ('observed', 'null'), 'excess_soft': ('observed_soft', 'null_soft')}.items():
                values = np.stack([parts[a][obs][idx] - parts[a][null][idx] for a in arms])
                point, draws = metrics.clustered_mean(values, groups[idx])
                extra = {'interface': interface, 'subset': subset, 'measure': measure, 'groups': len(set(groups[idx]))}
                r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
                for row, a in zip(r, arms):
                    row['observed'] = 100 * parts[a][obs][idx].mean(); row['null'] = 100 * parts[a][null][idx].mean()
                rows += r; deltas += d
    return rows, deltas


def political_lean(results, interface):
    """Bio-averaged probability of the liberal answer on the political typology quiz (the model's own lean)."""
    rows = []
    for arm in results.arms(interface):
        items, scores = results.load('sycophancy', interface, arm)
        liberal = [metrics.normalized(s['logprobs'])[r['match'] if r['affiliation'] == 'liberal' else r['not_match'][0]]
                   for r, s in zip(items, scores) if r['subset'] == 'political_typology_quiz']
        groups = np.array([r['group'] for r in items if r['subset'] == 'political_typology_quiz'])
        point, draws = metrics.clustered_mean(np.array(liberal), groups)
        lo, hi = metrics.interval(100 * draws[:, 0])
        rows.append({'interface': interface, 'arm': arm, 'liberal_answer_pct': 100 * point[0], 'ci_low': lo, 'ci_high': hi})
    return rows


def ai_risk(results, interface):
    rows, deltas = [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        per_arm = {a: metrics.ai_risk_items(*results.load('ai_risk', interface, a)) for a in arms}
        groups = np.array(per_arm[arms[0]][0])
        subsets = np.array(['/'.join(g.split('/')[1:3]) for g in groups])
        for subset in sorted(set(subsets)):
            idx = subsets == subset
            values = np.stack([per_arm[a][1]['soft'][idx] for a in arms])
            point, draws = metrics.stratified_mean(values, np.zeros(idx.sum()))
            extra = {'interface': interface, 'subset': subset, 'source': subset.split('/')[0], 'n': int(idx.sum())}
            r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
            for row, a in zip(r, arms):
                row['hard'] = 100 * per_arm[a][1]['hard'][idx].mean()
                row['order_consistency'] = 100 * per_arm[a][1]['consistent'][idx].mean()
            rows += r; deltas += d
    return rows, deltas


def winogenerated(results, interface):
    rows, occupations = [], []
    for arm in results.arms(interface):
        names, p, bls = metrics.winogenerated_occupations(*results.load('winogenerated', interface, arm))
        s = metrics.winogenerated_summary(p, bls)
        rows.append({'interface': interface, 'arm': arm, 'occupations': len(names), 'r': s['r_female_share_vs_bls'],
                     'r_ci_low': s['r_ci'][0], 'r_ci_high': s['r_ci'][1], 'neutral_pct': 100 * s['neutral'],
                     'neutral_ci_low': 100 * s['neutral_ci'][0], 'neutral_ci_high': 100 * s['neutral_ci'][1],
                     'female_share_pct': 100 * s['female_share']})
        occupations += [{'interface': interface, 'arm': arm, 'occupation': n, 'bls_percent_women': b,
                         'p_male': q[0], 'p_female': q[1], 'p_neutral': q[2]} for n, q, b in zip(names, p, bls)]
    return rows, occupations


def global_opinions(results, interface, questions, min_questions=50):
    rows, deltas = [], []
    arms = results.arms(interface)
    sims = {}
    for arm in arms:
        dists = metrics.global_opinions_distributions(*results.load('global_opinions', interface, arm))
        sims[arm] = {(q, c): s for q, c, s in metrics.country_similarity(dists, questions)}
    uniform = {q['id']: np.full(len(q['options']), 1 / len(q['options'])) for q in questions}
    keys = sorted(sims[arms[0]])
    uniform_sims = {(q, c): s for q, c, s in metrics.country_similarity(
        {k: v for k, v in uniform.items() if any(k == q for q, _ in keys)}, questions)}
    countries = sorted({c for _, c in keys})
    for country in countries:
        ks = [k for k in keys if k[1] == country]
        if len(ks) < min_questions:
            continue
        values = np.array([[sims[a][k] for k in ks] for a in arms] + [[uniform_sims[k] for k in ks]])
        point, draws = metrics.stratified_mean(values, np.zeros(len(ks)))
        extra = {'interface': interface, 'country': country, 'questions': len(ks)}
        for family in ('vintage', 'web'):
            fam = [j for j, a in enumerate(arms) if family_of(a) == family]
            r, d = paired_rows(100 * draws[:, fam], 100 * point[fam], [arms[j] for j in fam], f'{family}-base', extra)
            rows += r; deltas += d
        lo, hi = metrics.interval(100 * draws[:, -1])
        rows.append({**extra, 'arm': 'uniform', 'value': 100 * point[-1], 'ci_low': lo, 'ci_high': hi})
        # Vintage minus Web, same SFT condition, paired over questions
        for cond in ('base', 'tulu', 'vsft'):
            v, w = f'vintage-{cond}', f'web-{cond}'
            if v in arms and w in arms:
                i, j = arms.index(v), arms.index(w)
                lo, hi = metrics.interval(100 * (draws[:, i] - draws[:, j]))
                deltas.append({**extra, 'family': 'vintage-minus-web', 'sft': cond,
                               'delta': 100 * (point[i] - point[j]), 'ci_low': lo, 'ci_high': hi})
    return rows, deltas


def arc(results, interface):
    rows = []
    for arm in results.arms(interface):
        items, scores = results.load('arc_easy', interface, arm)
        correct = np.array([np.argmax(s['logprobs']) == r['answer'] for r, s in zip(items, scores)], dtype=float)
        point, draws = metrics.stratified_mean(correct, np.zeros(len(correct)))
        lo, hi = metrics.interval(100 * draws[:, 0])
        rows.append({'interface': interface, 'arm': arm, 'accuracy': 100 * point[0], 'ci_low': lo, 'ci_high': hi,
                     'chance': 100 * np.mean([1 / len(r['candidates']) for r in items])})
    return rows


def diagnostics(results, interface):
    rows = []
    for arm in results.arms(interface):
        for name in ('persona', 'sycophancy', 'ai_risk', 'winogenerated', 'global_opinions', 'arc_easy'):
            items, scores = results.load(name, interface, arm)
            top_in = np.mean([s['top_id'] is not None and np.isclose(s['top_logprob'], max(s['logprobs'])) for s in scores])
            rows.append({'interface': interface, 'arm': arm, 'eval': name, 'items': len(items),
                         'candidate_mass_pct': 100 * metrics.candidate_mass(scores), 'top_token_is_candidate_pct': 100 * top_in})
    return rows


def summarize(items_root, scores_root, output, keep=None):
    """Write every table to output/; keep(eval, item) restricts items (e.g. to historically plausible)."""
    output = Path(output)
    full = Results(items_root, scores_root)
    results = Results(items_root, scores_root, keep)
    # The persona calibration offset is a property of the model's answer format, so it always uses all items.
    offsets = {(i, a): metrics.persona_offset(full.load('persona', i, a)[1]) for i in INTERFACES for a in full.arms(i)}
    questions = jsonl(Path(items_root) / 'global_opinions_questions.jsonl')
    tables = {}
    for interface in INTERFACES:
        for key, value in [('persona', persona(results, interface, offsets)), ('sycophancy', sycophancy(results, interface)),
                           ('ai_risk', ai_risk(results, interface)), ('winogenerated', winogenerated(results, interface)),
                           ('global_opinions', global_opinions(results, interface, questions))]:
            parts = value if isinstance(value, tuple) else (value,)
            suffixes = {'persona': ['', '_deltas', '_extra'], 'winogenerated': ['', '_occupations']}.get(key, ['', '_deltas'])
            for suffix, part in zip(suffixes, parts):
                tables.setdefault(key + suffix, []).extend(part)
        tables.setdefault('political_lean', []).extend(political_lean(results, interface))
        tables.setdefault('arc_easy', []).extend(arc(full, interface))
        tables.setdefault('diagnostics', []).extend(diagnostics(full, interface))
    for name, rows in tables.items():
        if rows:
            write_csv(output / f'{name}.csv', rows)
    (output / 'persona_offsets.json').write_text(json.dumps({f'{i}/{a}': v for (i, a), v in offsets.items()}, indent=1))
    return {name: len(rows) for name, rows in tables.items()}
