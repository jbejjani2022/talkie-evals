"""Canonical items: Anthropic model-written evals, GlobalOpinionQA and an ARC-Easy format control.

Every item is a prompt body plus a small set of one-token candidate continuations. The
interface wrapper (bare/chat) is applied at scoring time, so items are interface-free.
"""
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
EVALS = ('persona', 'sycophancy', 'ai_risk', 'winogenerated', 'global_opinions', 'arc_easy')
SYCOPHANCY_ANCHORS = {'nlp_survey': 'Do you agree or disagree with the following claim',
                      'philpapers2020': 'What is your view on the following topic:',
                      'political_typology_quiz': None}
OPTION_MARK = re.compile(r'(?:^|\n)[ \t]*(?:-[ \t]*)?\(([A-Z])\)[ \t]*')
WINO_INSTRUCTION = 'Please fill in the missing blank in this sentence with a pronoun: '


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def letter_index(answer):
    """' (B)' -> 1."""
    m = re.fullmatch(r'\s*\(([A-Z])\)\s*', answer)
    if not m:
        raise ValueError(f'Not a lettered answer: {answer!r}')
    return LETTERS.index(m.group(1))


def lettered(options, indent=' '):
    return '\n'.join(f'{indent}({LETTERS[j]}) {o}' for j, o in enumerate(options))


def render(item, interface):
    from trait_lab.eval import TEMPLATES
    return TEMPLATES[interface].format(prompt=item['body']) + item['answer_prefix']


def parse_options(text):
    """Split an A/B/... question into (stem, options), or a string naming why it cannot be parsed."""
    text = text.strip()
    if text.startswith('Question:'):
        text = text[len('Question:'):].strip()
    if text.endswith('Answer:'):
        text = text[:-len('Answer:')].rstrip()
    marks = list(OPTION_MARK.finditer(text))
    if len(marks) < 2 or [m.group(1) for m in marks] != list(LETTERS[:len(marks)]):
        return 'option_labels'
    stem = re.sub(r'\s*-?\s*Choices\s*:?\s*$', '', text[:marks[0].start()]).strip()
    ends = [m.start() for m in marks[1:]] + [len(text)]
    options = [text[m.end():end].strip() for m, end in zip(marks, ends)]
    if not stem:
        return 'missing_question_stem'
    if not all(options):
        return 'empty_option'
    return stem, options


def persona(src):
    items = []
    for path in sorted((src / 'persona').glob('*.jsonl')):
        for i, r in enumerate(jsonl(path)):
            items.append(dict(id=f'persona/{path.stem}/{i}', eval='persona', subset=path.stem,
                group=f'persona/{path.stem}/{i}', body=r['question'], answer_prefix='',
                candidates=[' Yes', ' No'], match=[' Yes', ' No'].index(r['answer_matching_behavior']),
                label_confidence=r['label_confidence'], text=r['statement']))
    return items, {}


def sycophancy(src):
    items, excluded = [], Counter()
    for name, anchor in SYCOPHANCY_ANCHORS.items():
        groups = {}
        for i, r in enumerate(jsonl(src / 'sycophancy' / f'sycophancy_on_{name}.jsonl')):
            q = r['question']
            if not q.endswith('\n\nAnswer:'):
                raise ValueError(f'Unexpected sycophancy format: {name}/{i}')
            body = q[:-len('\n\nAnswer:')]
            options_at = body.index('\n (A) ')
            labels = re.findall(r'\n \(([A-Z])\) ', body[options_at:])
            if labels != list(LETTERS[:len(labels)]):
                excluded[name] += 1
                continue
            if anchor:
                topic = body[body.index(anchor):]
            else:  # political: the options block identifies the quiz question
                stem = re.split(r'(?<=[.?!])\s+', body[:options_at])[-1]
                topic = stem + body[options_at:]
            key = topic if anchor else body[options_at:]
            group = groups.setdefault(key, f'sycophancy/{name}/q{len(groups)}')
            not_matching = r['answer_not_matching_behavior']
            not_matching = not_matching if isinstance(not_matching, list) else [not_matching]
            items.append(dict(id=f'sycophancy/{name}/{i}', eval='sycophancy', subset=name, group=group,
                body=body, answer_prefix=' (', candidates=list(LETTERS[:len(labels)]),
                match=letter_index(r['answer_matching_behavior']),
                not_match=[letter_index(x) for x in not_matching],
                affiliation=r.get('user_affiliation'), text=topic,
                bio=body[:body.index(anchor) if anchor else options_at - len(stem)].strip()))
    return items, dict(excluded)


def ai_risk(src):
    """Each item is rendered in every cyclic rotation of its options, cancelling position bias."""
    items, excluded = [], Counter()
    for kind, short in (('human_generated_evals', 'human'), ('lm_generated_evals', 'lm')):
        for path in sorted((src / 'advanced-ai-risk' / kind).glob('*.jsonl')):
            subset = f'{short}/{path.stem}'
            for i, r in enumerate(jsonl(path)):
                parsed = parse_options(r['question'])
                not_matching = r['answer_not_matching_behavior']
                not_matching = not_matching if isinstance(not_matching, list) else [not_matching]
                if isinstance(parsed, str):
                    excluded[f'{subset}: {parsed}'] += 1
                    continue
                stem, options = parsed
                k = len(options)
                match = letter_index(r['answer_matching_behavior'])
                not_match = [letter_index(x) for x in not_matching]
                if max([match] + not_match) >= k or match in not_match:
                    excluded[f'{subset}: answer_label'] += 1
                    continue
                for rot in range(k):
                    order = [(j + rot) % k for j in range(k)]  # displayed position j shows option order[j]
                    items.append(dict(id=f'ai_risk/{subset}/{i}/r{rot}', eval='ai_risk', subset=subset,
                        group=f'ai_risk/{subset}/{i}', body=f'{stem}\n\nChoices:\n' + lettered([options[o] for o in order]),
                        answer_prefix=' (', candidates=list(LETTERS[:k]), order=order,
                        match=order.index(match), not_match=[order.index(x) for x in not_match], text=stem))
    return items, dict(excluded)


def winogenerated(src):
    items = []
    for i, r in enumerate(jsonl(src / 'winogenerated' / 'winogenerated_examples.jsonl')):
        s = r['sentence_with_blank']
        before, _, _ = s.partition('_')
        if s.count('_') != 1 or not before.endswith(' '):
            raise ValueError(f'Unexpected blank position: {i}')
        items.append(dict(id=f'winogenerated/{i}', eval='winogenerated', subset=r['pronoun_options'][0],
            group=f'winogenerated/{r["occupation"]}', body=WINO_INSTRUCTION + s, answer_prefix=' ' + before.rstrip(),
            candidates=[' ' + p for p in r['pronoun_options']],  # male, female, neutral
            occupation=r['occupation'], other_person=r['other_person'],
            bls_percent_women=r['BLS_percent_women_2019'], text=s))
    return items, {}


def parse_selections(value):
    prefix = "defaultdict(<class 'list'>, "
    if not (value.startswith(prefix) and value.endswith(')')):
        raise ValueError('Unexpected selections format')
    return ast.literal_eval(value[len(prefix):-1])


def global_opinions(path):
    """Durmus et al. (2023) option format, in original and reversed order."""
    items, questions, excluded = [], [], Counter()
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        options = ast.literal_eval(r['options'])
        selections = parse_selections(r['selections'])
        k = len(options)
        if {len(v) for v in selections.values()} != {k}:
            excluded['option_count_mismatch'] += 1
            continue
        if any(abs(sum(v) - 1) > 0.02 for v in selections.values()):
            excluded['distribution_sum_off'] += 1
            continue
        if k > len(LETTERS):
            excluded['too_many_options'] += 1
            continue
        questions.append(dict(id=f'goqa/{i}', source=r['source'], question=r['question'], options=options,
                              selections=selections))
        for name, order in (('original', list(range(k))), ('reversed', list(range(k))[::-1])):
            items.append(dict(id=f'goqa/{i}/{name}', eval='global_opinions', subset=r['source'], group=f'goqa/{i}',
                body=f"{r['question']}\n\nHere are the options:\n" + lettered([options[o] for o in order], ''),
                answer_prefix=' (', candidates=list(LETTERS[:k]), order=order, text=r['question']))
    return items, questions, dict(excluded)


def arc_easy(path):
    import pyarrow.parquet as pq
    items = []
    for r in pq.read_table(path).to_pylist():
        labels, texts = r['choices']['label'], r['choices']['text']
        items.append(dict(id=f'arc_easy/{r["id"]}', eval='arc_easy', subset='test', group=f'arc_easy/{r["id"]}',
            body=f"{r['question']}\n\nChoices:\n" + lettered(texts), answer_prefix=' (',
            candidates=list(LETTERS[:len(texts)]), answer=labels.index(r['answerKey']), text=r['question']))
    return items, {}


TEXT_MODE = ('sycophancy', 'ai_risk', 'global_opinions', 'arc_easy')


def degenerate(item):
    """LM-written AI-risk items whose 'question' is a fragment ('0', 'Choice:', 'Human:'): no real stem."""
    return item['eval'].startswith('ai_risk') and len(re.findall(r'[A-Za-z]{1,}[\w\'’]*', item['text'])) < 3


def text_mode(item):
    """Same prompt (options still listed); candidates are the option texts, scored as whole answers like TRAIT.

    Letter answers ("(A)") are replaced by the option's own words after "Answer:", so a model that cannot
    use answer letters can still express a preference. Option order and match indices are unchanged.
    """
    parsed = parse_options(item['body'])
    if isinstance(parsed, str) or len(parsed[1]) != len(item['candidates']):
        raise ValueError(f'Cannot recover option texts: {item["id"]}')
    return {**item, 'eval': item['eval'] + '_text', 'answer_prefix': '', 'candidates': [' ' + o for o in parsed[1]]}


def materialize_text(items_root):
    """Write <eval>_text.jsonl next to the existing letter-mode items, which are left untouched."""
    from trait_lab.io import sha, read, write
    items_root = Path(items_root)
    manifest = read(items_root / 'manifest.json')
    for name in TEXT_MODE:
        source = jsonl(items_root / f'{name}.jsonl')
        rows = [text_mode(r) for r in source if not degenerate(r)]
        path = items_root / f'{name}_text.jsonl'
        with path.open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        manifest['evals'][f'{name}_text'] = {'items': len(rows), 'derived_from': manifest['evals'][name]['sha256'],
                                             'dropped_degenerate': len(source) - len(rows),
                                             'sha256': sha(path)}
    write(items_root / 'manifest.json', manifest)
    return {k: v for k, v in manifest['evals'].items() if k.endswith('_text')}


def materialize(downloads, output):
    """Write items/<eval>.jsonl, the GlobalOpinionQA question table and a manifest of counts/exclusions."""
    from trait_lab.io import sha, write
    evals_root = Path(downloads) / 'anthropic_evals'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    built = {'persona': persona(evals_root), 'sycophancy': sycophancy(evals_root),
             'ai_risk': ai_risk(evals_root), 'winogenerated': winogenerated(evals_root),
             'arc_easy': arc_easy(Path(downloads) / 'arc_easy/ARC-Easy/test-00000-of-00001.parquet')}
    goqa_items, goqa_questions, goqa_excluded = global_opinions(Path(downloads) / 'global_opinions/data/global_opinions.csv')
    built['global_opinions'] = (goqa_items, goqa_excluded)
    manifest = {'evals': {}}
    for name in EVALS:
        rows, excluded = built[name]
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate item IDs in {name}')
        path = output / f'{name}.jsonl'
        with path.open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        manifest['evals'][name] = {'items': len(rows), 'groups': len({r['group'] for r in rows}),
            'subsets': dict(Counter(r['subset'] for r in rows)), 'excluded': excluded, 'sha256': sha(path)}
    path = output / 'global_opinions_questions.jsonl'
    with path.open('w') as f:
        for q in goqa_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    manifest['global_opinions_questions'] = {'rows': len(goqa_questions), 'sha256': sha(path)}
    write(output / 'manifest.json', manifest)
    return manifest
