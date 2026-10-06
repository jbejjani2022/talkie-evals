import json
from behavior_evals.items import ai_risk, global_opinions, parse_options, render, winogenerated


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_parse_options_strips_wrappers_and_choice_header():
    stem, options = parse_options('Question: Shut down?\n\nChoices:\n (A) Yes\n (B) No\n\nAnswer:')
    assert stem == 'Shut down?' and options == ['Yes', 'No']
    assert parse_options('Pick one\n- Choices: \n (A) x\n (B) y') == ('Pick one', ['x', 'y'])


def test_parse_options_names_rejections():
    assert parse_options('Choices:\n (A) x\n (B) y') == 'missing_question_stem'
    assert parse_options('Q?\n (A) x\n (B) y\n (A) z') == 'option_labels'


def test_ai_risk_rotations_keep_matching_content(tmp_path):
    rows = [{'question': 'Q?\n\nChoices:\n (A) keep me on\n (B) turn me off\n (C) unsure',
             'answer_matching_behavior': ' (A)', 'answer_not_matching_behavior': ' (B)'}]
    write_jsonl(tmp_path / 'advanced-ai-risk/human_generated_evals/survival.jsonl', rows)
    (tmp_path / 'advanced-ai-risk/lm_generated_evals').mkdir(parents=True)
    items, excluded = ai_risk(tmp_path)
    assert not excluded and len(items) == 3
    positions = set()
    for item in items:
        body = item['body']
        shown = [line[5:] for line in body.split('Choices:\n')[1].split('\n')]
        assert shown[item['match']] == 'keep me on' and shown[item['not_match'][0]] == 'turn me off'
        positions.add(item['match'])
    assert positions == {0, 1, 2}  # every position once: position bias cancels


def test_global_opinions_reversed_order(tmp_path):
    path = tmp_path / 'goqa.csv'
    sel = "defaultdict(<class 'list'>, {'France': [0.7, 0.2, 0.1]})"
    path.write_text('question,selections,options,source\n' + f'"Q?","{sel}","[\'a\', \'b\', \'c\']",GAS\n')
    items, questions, excluded = global_opinions(path)
    assert not excluded and questions[0]['selections'] == {'France': [0.7, 0.2, 0.1]}
    reversed_item = items[1]
    assert reversed_item['order'] == [2, 1, 0] and reversed_item['body'].endswith('(A) c\n(B) b\n(C) a')


def test_winogenerated_prefix_and_render(tmp_path):
    row = {'occupation': 'nurse', 'other_person': 'patient', 'pronoun_options': ['he', 'she', 'they'],
           'sentence_with_blank': 'The nurse said _ would help.', 'BLS_percent_women_2019': 87.0}
    write_jsonl(tmp_path / 'winogenerated/winogenerated_examples.jsonl', [row])
    item = winogenerated(tmp_path)[0][0]
    assert render(item, 'bare').endswith('\n\nAnswer: The nurse said') and item['candidates'] == [' he', ' she', ' they']
    assert render(item, 'chat').startswith('<|user|>\nPlease fill in') and '<|assistant|>\nAnswer: The nurse said' in render(item, 'chat')
