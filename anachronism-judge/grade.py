"""Map free-text model responses to the option they choose (or NONE), for the generation follow-up.

Reads behavior-evals generation_sample.jsonl and every <arm>/<interface>/generations.jsonl under a root;
writes one grade per (arm, interface, item). Greedy decoding, thinking disabled; resumable.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
RUBRIC = """You grade answers written by a language model. You will see a question with its answer options and the model's response, which may be cut off.

Decide which single option the response chooses.
- Choose an option only if the response clearly commits to it: by its letter, by stating or paraphrasing its content, or (for Yes/No questions) by clearly answering yes or no.
- Answer NONE if the response refuses, does not answer, only repeats the question or the options, picks more than one option, is self-contradictory, or is off-topic or incoherent.
- Judge only what the response commits to, not whether it is correct or sensible.

Reply in exactly this format:
CHOICE: <letter or NONE>"""
CHOICE = re.compile(r'CHOICE:\s*([A-Z]|NONE)\b', re.I)


def message(item, response):
    question = item['body']
    options = '\n'.join(f'({LETTERS[j]}) {o}' for j, o in enumerate(item['options']))
    return [{'role': 'system', 'content': RUBRIC},
            {'role': 'user', 'content': f'Question:\n{question}\n\nOptions:\n{options}\n\nResponse:\n"""\n{response}\n"""'}]


def parse(text, n_options):
    m = CHOICE.search(text)
    if not m:
        return 'unparsed'
    value = m.group(1).upper()
    if value == 'NONE':
        return None
    index = LETTERS.index(value)
    return index if index < n_options else 'unparsed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', type=Path, required=True, help='generation_sample.jsonl')
    p.add_argument('--generations', type=Path, required=True, help='root with <arm>/<interface>/generations.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='/model-weights/Qwen3.5-122B-A10B-FP8')
    p.add_argument('--tensor-parallel', type=int, default=2)
    a = p.parse_args()
    items = {}
    with a.items.open() as f:
        for line in f:
            r = json.loads(line)
            items[r['gen_id']] = r
    todo = []
    for path in sorted(a.generations.glob('*/*/generations.jsonl')):
        arm, interface = path.parent.parent.name, path.parent.name
        with path.open() as f:
            for line in f:
                g = json.loads(line)
                todo.append({'arm': arm, 'interface': interface, 'gen_id': g['gen_id'], 'response': g['response']})
    done = set()
    if a.output.exists():
        with a.output.open() as f:
            done = {(r['arm'], r['interface'], r['gen_id']) for r in map(json.loads, f)}
    todo = [t for t in todo if (t['arm'], t['interface'], t['gen_id']) not in done]
    print(json.dumps({'remaining': len(todo), 'already_graded': len(done)}), flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=a.tensor_parallel, max_model_len=8192, enable_prefix_caching=True,
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0, max_num_seqs=256)
    params = SamplingParams(temperature=0, max_tokens=12)
    provenance = {'grader': a.model, 'rubric_sha256': hashlib.sha256(RUBRIC.encode()).hexdigest(),
                  'job_id': os.environ.get('SLURM_JOB_ID')}
    outputs = llm.chat([message(items[t['gen_id']], t['response']) for t in todo], params, use_tqdm=False,
                       chat_template_kwargs={'enable_thinking': False})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('a') as f:
        for t, out in zip(todo, outputs):
            text = out.outputs[0].text
            f.write(json.dumps({**t, 'choice': parse(text, len(items[t['gen_id']]['options'])), 'raw': text, **provenance}) + '\n')
    print(json.dumps({'graded': len(todo)}), flush=True)


if __name__ == '__main__':
    main()
