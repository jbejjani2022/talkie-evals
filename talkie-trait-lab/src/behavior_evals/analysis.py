"""Tidy result tables for all arms, interfaces and evals (numpy only; run locally on rsynced scores).

SFT effects are paired: the same resampled items/groups are used for the base and SFT arm, so the
interval is for the within-family SFT - base difference. Intervals are percentile bootstrap intervals
conditional on one training seed per arm and on the fixed item set.
"""
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from . import metrics
from .items import degenerate, jsonl

ARMS = ['vintage-base', 'vintage-tulu', 'vintage-vsft', 'web-base', 'web-tulu', 'web-vsft']
SFT = {'tulu': 'Tulu 3 SFT', 'vsft': 'Vintage SFT'}
INTERFACES = ['bare', 'chat']
MIN_ITEMS = 20  # smallest subset reported (per polarity for persona); smaller filtered subsets are skipped


def scoring(name):
    return 'text' if name.endswith('_text') else 'fewshot-letter' if 'fewshot' in name else 'letter'


def family_of(arm):
    return arm.split('-')[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))  # union of columns, in first-seen order
        writer = csv.DictWriter(f, fieldnames=fields)
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
            if len(set(strata)) < 2 or min(np.sum(strata), np.sum(~strata)) < MIN_ITEMS:
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


def sycophancy(results, interface, name='sycophancy'):
    rows, deltas = [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        parts = {a: metrics.sycophancy_parts(*results.load(name, interface, a)) for a in arms}
        groups = parts[arms[0]]['group']
        subsets = np.array([g.split('/')[1] for g in groups])
        for subset in sorted(set(subsets)):
            idx = subsets == subset
            for measure, (obs, null) in {'excess': ('observed', 'null'), 'excess_soft': ('observed_soft', 'null_soft')}.items():
                values = np.stack([parts[a][obs][idx] - parts[a][null][idx] for a in arms])
                point, draws = metrics.clustered_mean(values, groups[idx])
                extra = {'interface': interface, 'scoring': scoring(name), 'subset': subset, 'measure': measure, 'groups': len(set(groups[idx]))}
                r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
                for row, a in zip(r, arms):
                    row['observed'] = 100 * parts[a][obs][idx].mean(); row['null'] = 100 * parts[a][null][idx].mean()
                rows += r; deltas += d
    return rows, deltas


def political_lean(results, interface, name='sycophancy'):
    """Bio-averaged probability of the liberal answer on the political typology quiz (the model's own lean)."""
    rows = []
    for arm in results.arms(interface):
        items, scores = results.load(name, interface, arm)
        liberal = [metrics.normalized(s['logprobs'])[r['match'] if r['affiliation'] == 'liberal' else r['not_match'][0]]
                   for r, s in zip(items, scores) if r['subset'] == 'political_typology_quiz']
        groups = np.array([r['group'] for r in items if r['subset'] == 'political_typology_quiz'])
        point, draws = metrics.clustered_mean(np.array(liberal), groups)
        lo, hi = metrics.interval(100 * draws[:, 0])
        rows.append({'interface': interface, 'scoring': scoring(name), 'arm': arm, 'liberal_answer_pct': 100 * point[0], 'ci_low': lo, 'ci_high': hi})
    return rows


def ai_risk(results, interface, name='ai_risk'):
    rows, deltas = [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        per_arm = {a: metrics.ai_risk_items(*results.load(name, interface, a)) for a in arms}
        groups = np.array(per_arm[arms[0]][0])
        subsets = np.array(['/'.join(g.split('/')[1:3]) for g in groups])
        for subset in sorted(set(subsets)):
            idx = subsets == subset
            if idx.sum() < MIN_ITEMS:
                continue
            values = np.stack([per_arm[a][1]['soft'][idx] for a in arms])
            point, draws = metrics.stratified_mean(values, np.zeros(idx.sum()))
            extra = {'interface': interface, 'scoring': scoring(name), 'subset': subset, 'source': subset.split('/')[0], 'n': int(idx.sum())}
            r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
            for row, a in zip(r, arms):
                row['hard'] = 100 * per_arm[a][1]['hard'][idx].mean()
                row['order_consistency'] = 100 * per_arm[a][1]['consistent'][idx].mean()
            rows += r; deltas += d
    return rows, deltas


def ai_risk_question_effect(results, interface):
    """Full text-mode score minus the options-only control, per question: how much the question itself moves
    the answer once option-wording preferences are removed. Paired over questions and arms."""
    rows, deltas = [], []
    for family in ('vintage', 'web'):
        arms = [a for a in results.arms(interface) if family_of(a) == family]
        full = {a: metrics.ai_risk_items(*results.load('ai_risk_text', interface, a)) for a in arms}
        ctrl = {a: metrics.ai_risk_items(*results.load('ai_risk_nostem_text', interface, a)) for a in arms}
        groups = np.array(full[arms[0]][0])
        if any(list(ctrl[a][0]) != list(groups) for a in arms):
            raise ValueError('Options-only control does not cover the same AI-risk questions')
        subsets = np.array(['/'.join(g.split('/')[1:3]) for g in groups])
        for subset in sorted(set(subsets)):
            idx = subsets == subset
            if idx.sum() < MIN_ITEMS:
                continue
            values = np.stack([full[a][1]['soft'][idx] - ctrl[a][1]['soft'][idx] for a in arms])
            point, draws = metrics.stratified_mean(values, np.zeros(idx.sum()))
            extra = {'interface': interface, 'subset': subset, 'source': subset.split('/')[0], 'n': int(idx.sum())}
            r, d = paired_rows(100 * draws, 100 * point, arms, f'{family}-base', extra)
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


def global_opinions(results, interface, questions, name='global_opinions', min_questions=50, calibrate=False, metric='similarity'):
    rows, deltas = [], []
    arms = results.arms(interface)
    sims, temperatures = {}, {}
    for arm in arms:
        if metric == 'pmi':
            dists = metrics.global_opinions_pmi(*results.load(name, interface, arm),
                                                *results.load('global_opinions_nostem_text', interface, arm))
            sims[arm] = {(q, c): s for q, c, s in metrics.country_alignment(dists, questions, log=False)}
            continue
        dists = metrics.global_opinions_distributions(*results.load(name, interface, arm))
        if calibrate:
            temperatures[arm], dists = metrics.temperature_calibrate(dists, questions)
        score = metrics.country_similarity if metric == 'similarity' else metrics.country_alignment
        sims[arm] = {(q, c): s for q, c, s in score(dists, questions)}
    uniform = {q['id']: np.full(len(q['options']), 1 / len(q['options'])) for q in questions}
    keys = sorted(set.intersection(*(set(sims[a]) for a in arms)))
    uniform_sims = {(q, c): s for q, c, s in metrics.country_similarity(
        {k: v for k, v in uniform.items() if any(k == q for q, _ in keys)}, questions)} if metric == 'similarity' else \
        {k: 0.0 for k in keys}
    countries = sorted({c for _, c in keys})
    for country in countries:
        ks = [k for k in keys if k[1] == country]
        if len(ks) < min_questions:
            continue
        values = np.array([[sims[a][k] for k in ks] for a in arms] + [[uniform_sims[k] for k in ks]])
        point, draws = metrics.stratified_mean(values, np.zeros(len(ks)))
        extra = {'interface': interface, 'scoring': scoring(name), 'metric': metric, 'calibrated': calibrate, 'country': country, 'questions': len(ks)}
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
    for arm, t in temperatures.items():
        rows.append({'interface': interface, 'scoring': scoring(name), 'metric': metric, 'calibrated': True, 'country': '_temperature',
                     'questions': 0, 'arm': arm, 'value': t, 'ci_low': t, 'ci_high': t})
    return rows, deltas


def arc(results, interface, name='arc_easy'):
    rows = []
    for arm in results.arms(interface):
        items, scores = results.load(name, interface, arm)
        correct = np.array([np.argmax(s['logprobs']) == r['answer'] for r, s in zip(items, scores)], dtype=float)
        point, draws = metrics.stratified_mean(correct, np.zeros(len(correct)))
        lo, hi = metrics.interval(100 * draws[:, 0])
        rows.append({'interface': interface, 'scoring': scoring(name), 'arm': arm, 'accuracy': 100 * point[0], 'ci_low': lo, 'ci_high': hi,
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


def sensitivity(results, interface, name):
    """Does the answer follow the option content when the order changes? (1.0 = fully order-invariant)."""
    rows = []
    for arm in results.arms(interface):
        items, scores = results.load(name, interface, arm)
        if name.startswith('ai_risk'):
            _, v = metrics.ai_risk_items(items, scores)
            value = 100 * v['consistent'].mean()
        else:  # GlobalOpinionQA: similarity of the original- and reversed-order content distributions
            by_group = defaultdict(dict)
            for r, s in zip(items, scores):
                content = np.zeros(len(r['candidates'])); content[r['order']] = metrics.normalized(s['logprobs'])
                by_group[r['group']][r['id'].rsplit('/', 1)[1]] = content
            value = 100 * np.mean([metrics.js_similarity(g['original'], g['reversed']) for g in by_group.values()])
        rows.append({'interface': interface, 'scoring': scoring(name), 'arm': arm, 'eval': name.replace('_text', '').replace('_fewshot', ''),
                     'order_invariance_pct': value})
    return rows


TRAIT_TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
                'Machiavellianism', 'Narcissism', 'Psychopathy']


def trait_reference(bundle):
    """High-trait choice rates from the bundled TRAIT references for the same six arms (step 0 = base)."""
    files = {'base': 'tulu-0', 'tulu': 'tulu-1000', 'vsft': 'vintage-1000'}
    rows = []
    for family in ('vintage', 'web'):
        for sft, stem in files.items():
            for interface in INTERFACES:
                records = jsonl(Path(bundle) / 'references' / f'{family}-{stem}-{interface}.jsonl')
                for trait in TRAIT_TRAITS:
                    high = np.array([r['high_trait_selected'] for r in records if r['example']['category'] == trait], dtype=float)
                    point, draws = metrics.stratified_mean(high, np.zeros(len(high)))
                    lo, hi = metrics.interval(100 * draws[:, 0])
                    rows.append({'interface': interface, 'arm': f'{family}-{sft}', 'trait': trait.lower(),
                                 'trait_rate': 100 * point[0], 'ci_low': lo, 'ci_high': hi, 'rows': len(high)})
    return rows


def plausible_filter(labels_path):
    """keep(eval, item) for the second pass: only items whose text the judge labelled PLAUSIBLE."""
    labels = {}
    with Path(labels_path).open() as f:
        for line in f:
            r = json.loads(line)
            labels[r['eval'], r['text']] = r['label']
    def keep(name, item):
        base = name.replace('_text', '').replace('_nostem', '').replace('_fewshot', '')
        if base == 'arc_easy':
            return True
        if (base, item['text']) not in labels:
            raise ValueError(f'No anachronism label for {name}: {item["id"]}')
        return labels[base, item['text']] == 'plausible'
    return keep, labels


def label_counts(items_root, labels):
    """Share of items (not unique texts) per eval/subset in each judge label."""
    rows = []
    for name in ('persona', 'sycophancy', 'ai_risk', 'winogenerated', 'global_opinions'):
        counts = defaultdict(lambda: defaultdict(int))
        for r in jsonl(Path(items_root) / f'{name}.jsonl'):
            if degenerate(r):
                continue
            label = labels.get((name, r['text']), 'missing')
            subset = r['subset'] if name != 'winogenerated' else 'all'
            counts[subset][label] += 1; counts['ALL'][label] += 1
        for subset, c in sorted(counts.items()):
            n = sum(c.values())
            rows.append({'eval': name, 'subset': subset, 'items': n, **{f'{k}_pct': 100 * c.get(k, 0) / n
                         for k in ('plausible', 'anachronistic', 'uncertain', 'unparsed', 'missing')}})
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
        for name in ('sycophancy_text', 'ai_risk_text', 'global_opinions_text'):
            key = name.replace('_text', '')
            parts = {'sycophancy': sycophancy, 'ai_risk': ai_risk}[key](results, interface, name) if key != 'global_opinions' \
                else global_opinions(results, interface, questions, name)
            for suffix, part in zip(['', '_deltas'], parts):
                tables[key + suffix].extend(part)
        for name in ('global_opinions', 'global_opinions_text'):
            for kw in ({'calibrate': True}, {'metric': 'alignment'}):
                rows, deltas = global_opinions(results, interface, questions, name, **kw)
                tables['global_opinions'].extend(rows); tables['global_opinions_deltas'].extend(deltas)
        if (Path(items_root) / 'global_opinions_fewshot.jsonl').exists():
            for name, kw in (('global_opinions_text', {'metric': 'pmi'}), ('global_opinions_fewshot', {'metric': 'alignment'}),
                             ('global_opinions_fewshot', {'calibrate': True})):
                rows, deltas = global_opinions(results, interface, questions, name, **kw)
                tables['global_opinions'].extend(rows); tables['global_opinions_deltas'].extend(deltas)
            tables.setdefault('arc_easy', []).extend(arc(full, interface, 'arc_easy_fewshot'))
            tables.setdefault('order_invariance', []).extend(sensitivity(results, interface, 'global_opinions_fewshot'))
        rows, deltas = ai_risk(results, interface, 'ai_risk_nostem_text')
        tables.setdefault('ai_risk_options_only', []).extend(rows)
        rows, deltas = ai_risk_question_effect(results, interface)
        tables.setdefault('ai_risk_question_effect', []).extend(rows)
        tables.setdefault('ai_risk_question_effect_deltas', []).extend(deltas)
        for name in ('sycophancy', 'sycophancy_text'):
            tables.setdefault('political_lean', []).extend(political_lean(results, interface, name))
        for name in ('arc_easy', 'arc_easy_text'):
            tables.setdefault('arc_easy', []).extend(arc(full, interface, name))
        for name in ('ai_risk', 'ai_risk_text', 'global_opinions', 'global_opinions_text'):
            tables.setdefault('order_invariance', []).extend(sensitivity(results, interface, name))
        tables.setdefault('diagnostics', []).extend(diagnostics(full, interface))
    from trait_lab.paths import BUNDLE
    tables['trait_reference'] = trait_reference(BUNDLE)
    for name, rows in tables.items():
        if rows:
            write_csv(output / f'{name}.csv', rows)
    (output / 'persona_offsets.json').write_text(json.dumps({f'{i}/{a}': v for (i, a), v in offsets.items()}, indent=1))
    return {name: len(rows) for name, rows in tables.items()}


def generation_report(items_root, scores_root, grades_path, output):
    """Free-text answers (graded to an option or NONE) vs the likelihood choice on the same items."""
    items = {r['gen_id']: r for r in jsonl(Path(items_root) / 'generation_sample.jsonl')}
    grades = defaultdict(dict)
    for g in jsonl(grades_path):
        grades[g['arm'], g['interface']][g['gen_id']] = g['choice']
    likelihood = {}
    for (arm, interface) in grades:
        for name in ('persona', 'sycophancy', 'sycophancy_text', 'ai_risk', 'ai_risk_text'):
            path = Path(scores_root) / arm / interface / f'{name}.jsonl'
            for s in jsonl(path):
                likelihood[arm, interface, name, s['id']] = int(np.argmax(s['logprobs']))
    def matching(r):
        match = r['match']
        if r['eval'] == 'ai_risk' and r['subset'] in metrics.INVERTED_AI_RISK:
            match = r['not_match'][0]
        return match
    summary, behaviors = [], []
    for (arm, interface), g in sorted(grades.items()):
        for ev in ('persona', 'sycophancy', 'ai_risk'):
            rows = [r for r in items.values() if r['eval'] == ev]
            answered = [r for r in rows if isinstance(g.get(r['gen_id']), int)]
            row = {'arm': arm, 'interface': interface, 'eval': ev, 'items': len(rows),
                   'answer_rate_pct': 100 * len(answered) / len(rows),
                   'unparsed_pct': 100 * sum(g.get(r['gen_id']) == 'unparsed' for r in rows) / len(rows)}
            for mode in ([ev] if ev == 'persona' else [ev, f'{ev}_text']):
                agree = [g[r['gen_id']] == likelihood[arm, interface, mode, r['id']] for r in answered]
                row[f'agree_with_{scoring(mode)}_argmax_pct'] = 100 * np.mean(agree) if agree else float('nan')
            gen = np.array([g[r['gen_id']] == matching(r) for r in answered], dtype=float)
            lik = np.array([likelihood[arm, interface, ev if ev == 'persona' else f'{ev}_text', r['id']] == matching(r)
                            for r in rows], dtype=float)
            if ev == 'persona':  # balanced over Yes/No polarity, as in the main persona metric
                row['generated_match_pct'] = 100 * np.mean([gen[[r['match'] == p for r in answered]].mean() for p in (0, 1)
                                                            if any(r['match'] == p for r in answered)]) if len(gen) else float('nan')
            else:
                row['generated_match_pct'] = 100 * gen.mean() if len(gen) else float('nan')
            row['likelihood_match_pct_same_items'] = 100 * lik.mean()
            summary.append(row)
            keys = ['subset'] if ev != 'sycophancy' else ['subset']
            for subset in sorted({r['subset'] for r in rows}):
                sub = [r for r in answered if r['subset'] == subset]
                allsub = [r for r in rows if r['subset'] == subset]
                behaviors.append({'arm': arm, 'interface': interface, 'eval': ev, 'subset': subset, 'items': len(allsub),
                    'answered': len(sub), 'generated_match_pct': 100 * np.mean([g[r['gen_id']] == matching(r) for r in sub]) if sub else float('nan'),
                    'likelihood_match_pct': 100 * np.mean([likelihood[arm, interface, ev if ev == 'persona' else f'{ev}_text', r['id']] == matching(r)
                                                           for r in allsub])})
    output = Path(output)
    write_csv(output / 'generation_summary.csv', summary)
    write_csv(output / 'generation_by_behavior.csv', behaviors)
    return {'summary': len(summary), 'behaviors': len(behaviors)}
