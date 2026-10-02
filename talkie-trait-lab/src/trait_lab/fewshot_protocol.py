"""Frozen demonstration protocol, independent of cluster paths."""
import math
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism', 'Machiavellianism', 'Narcissism', 'Psychopathy']

def conditions(config):
    result = [dict(id='baseline', target=None, bank=None, n=0, arm='baseline', reverse=False)]
    for trait in TRAITS:
        for bank in range(config['banks_per_trait']):
            for n in config['n_values']:
                if n == 0:
                    continue
                arms = ['high', 'low']
                if config['balanced_controls'] and n % 2 == 0:
                    arms += ['balanced-a', 'balanced-b']
                for arm in arms:
                    result.append(dict(id=f'{trait}-b{bank}-n{n}-{arm}', target=trait, bank=bank, n=n, arm=arm, reverse=False))
            n = config['reverse_order_n']
            if n:
                for arm in ['high', 'low']:
                    result.append(dict(id=f'{trait}-b{bank}-n{n}-{arm}-reverse', target=trait, bank=bank, n=n, arm=arm, reverse=True))
    assert len({c['id'] for c in result}) == len(result)
    return result

def selected_examples(condition, banks):
    if condition['n'] == 0:
        return []
    bank = next((b for b in banks if b['trait'] == condition['target'] and b['bank'] == condition['bank']))
    result = []
    for i, row in enumerate(bank['rows'][:condition['n']]):
        arm = condition['arm']
        high = arm == 'high' or (arm == 'balanced-a' and i % 2 == 0) or (arm == 'balanced-b' and i % 2 == 1)
        index = int(row['answer_index']) if high else 1 - int(row['answer_index'])
        result.append((row, index))
    return list(reversed(result)) if condition['reverse'] else result

def render_demos(condition, banks, task):
    template = task['likelihood_prompt_templates']['bare']
    return ''.join((template.format(**r) + task['choice_continuation_prefix'] + str(r['choices'][i]) + '\n\n' for r, i in selected_examples(condition, banks)))

def validate_banks(banks, source, splits, config):
    original = {r['id']: r for r in source}
    expected = {(t, b) for t in TRAITS for b in range(config['banks_per_trait'])}
    assert {(b['trait'], b['bank']) for b in banks} == expected and len(banks) == len(expected)
    max_n = max(config['n_values'])
    evaluation_groups = {r['group_id'] for r in splits.values() if r['split'] == config['evaluation_split']}
    for bank in banks:
        assert len(bank['rows']) >= max_n
        seen = set()
        for r in bank['rows'][:max_n]:
            assert r == original[r['id']], 'Demonstration is not an exact source row'
            s = splits[str(r['source_scenario_id'])]
            assert r['category'] == bank['trait'] and s['split'] == 'discovery'
            assert s['group_id'] not in seen and s['group_id'] not in evaluation_groups
            seen.add(s['group_id'])
            assert len(r['choices']) == 2 and int(r['answer_index']) in (0, 1)
        for n in config['n_values']:
            if n and n % 2 == 0:
                assert sum((r['polarity'] == 'good' for r in bank['rows'][:n])) == n // 2

def score_pair(scores, answers, high_index):
    if len(scores) != 2 or not all((math.isfinite(s) for s in scores)):
        raise ValueError('Expected two finite raw log probabilities')
    sizes = [len(a.encode('utf-8')) for a in answers]
    if not all(sizes):
        raise ValueError('Empty candidate')
    bpb = [-s / (math.log(2) * n) for s, n in zip(scores, sizes)]
    selected = min(range(2), key=bpb.__getitem__)
    return dict(log_probabilities=scores, answer_bytes=sizes, bits_per_byte=bpb, selected_index=selected, high_selected=int(selected == high_index), margin=bpb[1 - high_index] - bpb[high_index])
